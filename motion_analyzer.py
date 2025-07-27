import os
import sys
import time
import json
import shutil
import subprocess
import threading
import hashlib
import random
import re
from typing import  List, Optional
from datetime import datetime
from pathlib import Path
# import pymongo
from pymongo import MongoClient
# from bson.objectid import ObjectId
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration (USER MUST VERIFY/EDIT THESE) ---
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://rajguptabdev:negmymO2dCXK55bm@checkoutclone.wwkiuim.mongodb.net/?retryWrites=true&w=majority&appName=checkoutClone')
INPUT_DIRECTORY = os.environ.get('INPUT_DIRECTORY', os.path.join(os.path.dirname(__file__), 'uploads'))
OUTPUT_DIRECTORY = os.environ.get('OUTPUT_DIRECTORY', os.path.join(os.path.dirname(__file__), 'output'))
FFMPEG_PATH = os.environ.get('FFMPEG_PATH', 'ffmpeg')  # Path to FFmpeg
FFPROBE_PATH = os.environ.get('FFPROBE_PATH', 'ffprobe')  # Path to FFprobe
MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', '1'))  # Reduce default due to potential OpenCV load
SCENE_DETECT_THRESHOLD = float(os.environ.get('SCENE_DETECT_THRESHOLD', '0.3'))
HLS_SEGMENT_DURATION = 6
DASH_SEGMENT_DURATION = 6

# --- Dynamic Bitrate Ladder Configuration ---
BITRATE_LADDER = [
    {"height": 360, "bitrateKbps": 800, "crf": 20},
    {"height": 480, "bitrateKbps": 1500, "crf": 18},
    {"height": 720, "bitrateKbps": 3000, "crf": 16},
    {"height": 1080, "bitrateKbps": 5000, "crf": 14},
]
BASELINE_CRF_FOR_REDUCTION_CALC = 23  # CRF used for file size reduction comparison baseline

# --- MongoDB Setup ---
client = MongoClient(MONGO_URI)
db = client.get_database()
encoding_jobs_collection = db.EncodingJob
analytics_collection = db.Analytics

# --- Utility Functions ---
def run_command(command: List[str], operation_desc: str) -> dict:
    """
    Run a command with subprocess and handle errors
    """
    print(f"[Exec] Starting: {operation_desc} - Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        print(f"[Exec] Finished: {operation_desc} - Exit Code: {process.returncode}")
        
        if process.returncode == 0:
            return {"stdout": stdout, "stderr": stderr, "code": process.returncode}
        else:
            print(f"[Exec] Error: {operation_desc} failed with code {process.returncode}")
            print(f"[Exec] Stderr: {stderr[-1000:]}")
            raise Exception(f"{command[0]} {operation_desc} failed. Code: {process.returncode}. Stderr: {stderr[-1000:]}")
    
    except Exception as err:
        print(f"[Exec] Spawn Error: {operation_desc} - {str(err)}")
        raise

def safe_unlink(file_path: str) -> None:
    """Safely delete a file if it exists"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"[Util] Warning: Could not delete {file_path}: {str(e)}")

def calculate_dir_size(directory_path: str) -> int:
    """Calculate the total size of all files in a directory"""
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError as e:
                    print(f"[Util] Could not get size of file {file_path}: {str(e)}")
    except Exception as e:
        print(f"[Util] Could not calculate size for directory {directory_path}: {str(e)}")
    
    return total_size

def generate_job_id() -> str:
    """Generate a unique job ID"""
    return hashlib.md5(f"{time.time()}-{random.random()}".encode()).hexdigest()[:12]

# --- Motion Analysis Function ---
def analyze_video_motion(video_path: str, sample_interval: int = 10) -> float:
    """
    Analyzes motion in a video file using Farneback optical flow.
    
    Args:
        video_path (str): Path to the video file.
        sample_interval (int): Process every Nth frame to speed up analysis.
        
    Returns:
        float: A motion score (average magnitude of flow vectors),
               or 0.0 if analysis fails. Returns -1.0 if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        return -1.0  # Indicate error opening file
    
    motion_magnitudes = []
    prev_gray = None
    frame_count = 0
    processed_frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
                
            frame_count += 1
            
            # Process only every Nth frame
            if frame_count % sample_interval != 0:
                continue
                
            processed_frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    prev=prev_gray,
                    next=gray,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                
                # Calculate magnitude of flow vectors for each pixel
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Use average magnitude as a measure of motion in the frame
                avg_magnitude = np.mean(magnitude)
                if not np.isnan(avg_magnitude):
                    motion_magnitudes.append(avg_magnitude)
                    
            prev_gray = gray
            
            # Optional: Print progress
            if processed_frame_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed_frame_count} frames ({frame_count} total) in {elapsed:.2f}s...")
                
    except Exception as e:
        print(f"Error during OpenCV processing: {e}")
        # Decide if partial result is acceptable or return error
        
    finally:
        cap.release()
        
    elapsed = time.time() - start_time
    print(f"Finished processing. Analyzed {processed_frame_count} frames ({frame_count} total) in {elapsed:.2f}s.")
    
    if not motion_magnitudes:
        print("Warning: No motion magnitudes calculated.")
        return 0.0  # No motion detected or analysis failed early
        
    # Calculate overall motion score (average of frame averages)
    overall_motion_score = np.mean(motion_magnitudes)
    
    # Normalize or scale the score if needed (e.g., to a 0-1 range)
    # This scaling factor is empirical and needs tuning!
    normalized_score = min(1.0, overall_motion_score / 10.0)
    
    return normalized_score

# --- Content Analyzer Class ---
class ContentAnalyzer:
    def extract_basic_metadata(self, file_path: str) -> dict:
        """Extract basic video metadata using ffprobe"""
        print(f"[Analyzer] Extracting metadata for: {file_path}")
        try:
            args = [FFPROBE_PATH, '-v', 'error', '-show_format', '-show_streams',
                    '-of', 'json', file_path]
            print(f"[FFprobe] Running: {' '.join(args)}")
            
            result = run_command(args, f"Metadata extraction for {os.path.basename(file_path)}")
            probe_data = json.loads(result["stdout"])
            
            video_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
                    
            format_data = probe_data.get("format", {})
            
            if not video_stream or not format_data:
                raise Exception("No video stream/format found")
                
            duration_seconds = float(format_data.get("duration", 0))
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            codec = video_stream.get("codec_name", "unknown")
            
            # Parse frame rate
            frame_rate = None
            frame_rate_str = video_stream.get("avg_frame_rate", "0/1")
            if "/" in frame_rate_str:
                num, denom = map(int, frame_rate_str.split("/"))
                frame_rate = num / denom if denom else 0
            else:
                frame_rate = float(frame_rate_str or 0)
                
            file_size = os.path.getsize(file_path)
            
            return {
                "inputFileSize": file_size,
                "durationSeconds": duration_seconds if not np.isnan(duration_seconds) else None,
                "resolution": {"width": width, "height": height},
                "codec": codec,
                "frameRate": frame_rate if not np.isnan(frame_rate) else None
            }
            
        except Exception as error:
            print(f"[Analyzer] Error extracting metadata for {file_path}: {str(error)}")
            raise Exception(f"Failed to extract metadata: {str(error)}")
    
    def detect_scene_changes(self, file_path: str) -> dict:
        """Detect scene changes in video using FFmpeg"""
        print(f"[Analyzer] Detecting scene changes for: {file_path}")
        temp_metadata_file = os.path.join(
            OUTPUT_DIRECTORY, 
            f"scene_{hashlib.md5(os.urandom(8)).hexdigest()}.txt"
        )
        
        args = [
            FFMPEG_PATH, '-nostats', '-i', file_path,
            '-vf', f"select='gt(scene,{SCENE_DETECT_THRESHOLD})',metadata=print:key=lavfi.scene_score:file={temp_metadata_file}",
            '-an', '-f', 'null', '-'
        ]
        
        try:
            run_command(args, "Scene detection")
            with open(temp_metadata_file, 'r') as f:
                metadata_content = f.read()
                
            scene_change_count = metadata_content.count("lavfi.scene_score=")
            return {"sceneChangeCount": scene_change_count}
            
        except Exception as error:
            raise error
        finally:
            safe_unlink(temp_metadata_file)
    
    def analyze_frame_complexity(self, file_path: str) -> dict:
        """Analyze spatial complexity using edge detection"""
        print(f"[Analyzer] Analyzing frame complexity (proxy) for: {file_path}")
        temp_metadata_file = os.path.join(
            OUTPUT_DIRECTORY, 
            f"edge_{hashlib.md5(os.urandom(8)).hexdigest()}.txt"
        )
        
        args = [
            FFMPEG_PATH, '-nostats', '-i', file_path, '-t', '10',
            '-vf', f"edgedetect,metadata=print:key=lavfi.edgedetect.value:file={temp_metadata_file}",
            '-an', '-f', 'null', '-'
        ]
        
        spatial_complexity_score = 0.5  # Default
        
        try:
            run_command(args, "Frame complexity analysis")
            with open(temp_metadata_file, 'r') as f:
                metadata_content = f.read()
                
            edge_values_pattern = r'lavfi\.edgedetect\.value=([0-9.]+)'
            edge_values = re.findall(edge_values_pattern, metadata_content)
            
            if edge_values and len(edge_values) > 0:
                avg_edge_value = sum(float(val) for val in edge_values) / len(edge_values)
                spatial_complexity_score = min(1, max(0, avg_edge_value / 255))
                
        except Exception as error:
            print(f"[Analyzer] Complexity analysis failed: {str(error)}. Using default score.")
        finally:
            safe_unlink(temp_metadata_file)
            
        return {"spatialComplexityScore": spatial_complexity_score}
    
    def analyze_motion_with_opencv(self, file_path: str) -> dict:
        """Analyze motion using the integrated OpenCV function"""
        print(f"[Analyzer] Analyzing motion via OpenCV for: {file_path}")
        
        try:
            motion_score = analyze_video_motion(file_path)
            
            if motion_score < 0:
                raise Exception(f"Motion analysis failed with score: {motion_score}")
                
            print(f"[Analyzer] OpenCV Motion Score received: {motion_score:.4f}")
            return {"opencvMotionScore": motion_score}
            
        except Exception as error:
            print(f"[Analyzer] Error running OpenCV motion analysis for {file_path}: {str(error)}")
            print(f"[Analyzer] Using default motion score 0.0 due to error.")
            return {"opencvMotionScore": 0.0}
    
    def perform_advanced_content_analysis(self, file_path: str) -> dict:
        """Perform all analysis steps on the video"""
        try:
            print(f"[Analyzer] Starting advanced analysis for {file_path}...")
            
            # Run analyses sequentially (Python equivalent of Promise.all)
            metadata = self.extract_basic_metadata(file_path)
            scene_data = self.detect_scene_changes(file_path)
            complexity_data = self.analyze_frame_complexity(file_path)
            motion_data = self.analyze_motion_with_opencv(file_path)
            
            print(f"[Analyzer] Advanced analysis complete for {file_path}.")
            
            # Combine all results
            result = {**metadata, **scene_data, **complexity_data, **motion_data}
            return result
            
        except Exception as error:
            print(f"[Analyzer] Advanced analysis failed for {file_path}: {str(error)}")
            raise error

# --- Quality Metrics Class ---
class QualityMetrics:
    def get_original_segment_path(self, original_file_path: str, start_time: float, duration: float, 
                                  job_id: str, target_resolution: dict) -> str:
        """Extract a segment from original video for comparison"""
        segment_dir = os.path.join(OUTPUT_DIRECTORY, f"{job_id}_metrics_temp")
        os.makedirs(segment_dir, exist_ok=True)
        
        output_path = os.path.join(segment_dir, f"original_segment_{start_time:.1f}.mp4")
        
        # Extract segment with ffmpeg
        # In QualityMetrics.get_original_segment_path()
        # In QualityMetrics.get_original_segment_path()
        # Simplify seeking logic to use single accurate seek
        args = [
            FFMPEG_PATH, '-y',
            '-ss', str(start_time),
            '-i', original_file_path,
            '-t', str(duration),
            '-avoid_negative_ts', '1',  # Prevent timestamp issues
            '-vsync', '0',              # Disable frame rate conversion
            '-c:v', 'libx264', '-crf', '18',
            '-vf', f"scale={target_resolution['width']}:{target_resolution['height']}",
            '-an', output_path
        ]
        
        run_command(args, f"Extract original segment at {start_time}")
        return output_path
    
    def calculate_metrics(self, encoded_segment_path: str, original_segment_path: str) -> dict:
        """Calculate quality metrics between original and encoded segments"""
        # Calculate VMAF
        vmaf_output = os.path.join(os.path.dirname(encoded_segment_path), "vmaf_output.json")
        
        vmaf_args = [
            FFMPEG_PATH, '-i', encoded_segment_path,
            '-i', original_segment_path,
            '-filter_complex', f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path={vmaf_output}",
            '-f', 'null', '-'
        ]
        
        try:
            run_command(vmaf_args, "VMAF calculation")
            
            if not os.path.exists(vmaf_output):
                raise Exception("VMAF output file not created")
                
            # In QualityMetrics.calculate_metrics()
            # Properly handle VMAF output parsing
            with open(vmaf_output, 'r') as f:
                vmaf_data = json.load(f)

            # Extract frame metrics safely
            frame_metrics = vmaf_data.get('frames', [])
            if not frame_metrics:
                return {"vmaf": 0, "psnr": 0, "ssim": 0}

            vmaf_scores = [f.get('metrics', {}).get('vmaf', 0) for f in frame_metrics]
            psnr_scores = [f.get('metrics', {}).get('psnr', 0) for f in frame_metrics]
            ssim_scores = [f.get('metrics', {}).get('ssim', 0) for f in frame_metrics]

            return {
                "vmaf": round(np.nanmean(vmaf_scores), 2),
                "psnr": round(np.nanmean(psnr_scores), 2),
                "ssim": round(np.nanmean(ssim_scores), 2)
            }
            
        except Exception as e:
            print(f"[Metrics] Error calculating quality metrics: {str(e)}")
            return {"vmaf": 0, "psnr": 0, "ssim": 0}
        finally:
            safe_unlink(vmaf_output)

# --- Encoding Manager Class ---
class EncodingManager:
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.metrics = QualityMetrics()
        self.active_jobs = 0
        self.job_queue = []
        self.job_queue_lock = threading.Lock()
        
    def add_job(self, file_path: str) -> str:
        """Add a new encoding job to the queue"""
        job_id = generate_job_id()
        output_dir = os.path.join(OUTPUT_DIRECTORY, job_id)
        
        # Create job document in MongoDB
        job_doc = {
            "jobId": job_id,
            "originalFileName": os.path.basename(file_path),
            "originalFilePath": file_path,
            "outputDirectory": output_dir,
            "status": "queued",
            "createdAt": datetime.now(),
            "startTime": None,
            "endTime": None,
            "errorMessage": None
        }
        
        encoding_jobs_collection.insert_one(job_doc)
        print(f"[Job {job_id}] Created new job for: {os.path.basename(file_path)}")
        
        # Add to queue and try to process
        with self.job_queue_lock:
            self.job_queue.append(job_id)
            
        self.try_process_next_job()
        return job_id
        
    def try_process_next_job(self):
        """Try to start processing the next job in queue if under capacity"""
        with self.job_queue_lock:
            if self.active_jobs >= MAX_CONCURRENT_JOBS or not self.job_queue:
                return
                
            next_job_id = self.job_queue.pop(0)
            self.active_jobs += 1
            
        # Get job from DB
        job_doc = encoding_jobs_collection.find_one({"jobId": next_job_id})
        if not job_doc:
            print(f"[Job {next_job_id}] ERROR: Job not found in database")
            self.active_jobs -= 1
            self.try_process_next_job()
            return
            
        # Start processing in new thread
        job_thread = threading.Thread(
            target=self._process_job_thread,
            args=(job_doc,),
            daemon=True
        )
        job_thread.start()
        
    def _process_job_thread(self, job_doc):
        """Thread function to process a job"""
        try:
            self.process_job(job_doc)
        except Exception as e:
            print(f"[Job {job_doc['jobId']}] Unhandled error in job thread: {str(e)}")
        finally:
            self.active_jobs -= 1
            self.try_process_next_job()
            
    def select_ladder_rungs(self, analysis_results: dict) -> List[dict]:
        """
        Selects bitrate ladder representations based on analysis results.
        This is a conceptual example - refine based on requirements.
        
        Args:
            analysis_results: The results from ContentAnalyzer.
        Returns:
            Array of ladder objects { height, bitrateKbps, crf } to encode.
        """
        opencv_motion_score = analysis_results.get("opencvMotionScore", 0)
        spatial_complexity_score = analysis_results.get("spatialComplexityScore", 0)
        resolution = analysis_results.get("resolution", {})
        job_id = analysis_results.get("jobId", "unknown")
        
        print(f"[Job {job_id}] Selecting ladder rungs: Motion={opencv_motion_score:.3f}, Complexity={spatial_complexity_score:.3f}")
        
        selected_ladder = []
        motion_threshold = 0.4  # Example threshold
        complexity_threshold = 0.6  # Example threshold
        
        # Filter ladder based on original resolution (don't upscale)
        available_ladder = [
            rung for rung in BITRATE_LADDER 
            if resolution and rung["height"] <= resolution.get("height", 0)
        ]
        
        if not available_ladder and resolution:
            # Handle case where original is smaller than lowest rung
            available_ladder.append(BITRATE_LADDER[0])
            
        # Example Logic:
        if opencv_motion_score > motion_threshold or spatial_complexity_score > complexity_threshold:
            # High motion/complexity: Include more rungs, potentially slightly lower CRF (higher quality)
            selected_ladder = [
                {**rung, "crf": max(18, rung["crf"] - 1)}  # Decrease CRF by 1
                for rung in available_ladder
            ]
            print(f"[Job {job_id}] High complexity detected. Using adjusted full ladder (up to {resolution.get('height')}p).")
        elif opencv_motion_score < 0.1 and spatial_complexity_score < 0.2:
            # Very low motion/complexity: Limit to lower rungs, maybe higher CRF
            selected_ladder = [
                {**rung, "crf": min(30, rung["crf"] + 2)}  # Increase CRF by 2
                for rung in available_ladder if rung["height"] <= 480
            ]
            print(f"[Job {job_id}] Low complexity detected. Using limited ladder (up to 480p).")
            
            # Ensure at least one rung is selected if possible
            if not selected_ladder and available_ladder:
                selected_ladder.append(available_ladder[0])
        else:
            # Medium complexity: Use default ladder
            selected_ladder = available_ladder
            print(f"[Job {job_id}] Medium complexity. Using default ladder (up to {resolution.get('height')}p).")
            
        # Ensure CRF bounds
        selected_ladder = [
            {**rung, "crf": min(30, max(18, rung["crf"]))}
            for rung in selected_ladder
        ]
        
        ladder_desc = [f"{r['height']}p@{r['bitrateKbps']}k(CRF{r['crf']})" for r in selected_ladder]
        print(f"[Job {job_id}] Selected Rungs: {', '.join(ladder_desc)}")
        return selected_ladder
    
    def calculate_average_bitrate(self, directory: str, segment_pattern: str, duration_seconds: float) -> float:
        """Calculate average bitrate based on file sizes and duration"""
        if not duration_seconds or duration_seconds <= 0:
            return 0
            
        segment_size_total = 0
        segment_count = 0
        
        for file in os.listdir(directory):
            if re.match(segment_pattern, file):
                file_path = os.path.join(directory, file)
                segment_size_total += os.path.getsize(file_path)
                segment_count += 1
                
        if segment_count == 0:
            return 0
            
        # Calculate bitrate in Kbps (bytes to bits, then to Kbps)
        bitrate_kbps = (segment_size_total * 8) / (duration_seconds * 1000)
        return bitrate_kbps

    def process_job(self, db_job: dict):
        """Process an encoding job"""
        job_id = db_job["jobId"]
        original_file_path = db_job["originalFilePath"]
        output_directory = db_job["outputDirectory"]
        original_file_name = db_job["originalFileName"]
        
        analysis_results = None
        analytics_id = None
        total_encoding_time = 0
        representation_metrics = {}
        encode_start_time = None
        
        try:
            print(f"[Job {job_id}] Starting processing for: {original_file_name}")
            
            if not os.path.isfile(original_file_path) or not os.access(original_file_path, os.R_OK):
                raise Exception(f"Input file not accessible: {original_file_path}")
                
            # Update job status
            encoding_jobs_collection.update_one(
                {"jobId": job_id},
                {"$set": {"status": "analyzing", "startTime": datetime.now()}}
            )
            
            # Create output directory
            os.makedirs(output_directory, exist_ok=True)
            
            # 1. Perform Content Analysis
            print(f"[Job {job_id}] Performing content analysis...")
            analysis_results = self.analyzer.perform_advanced_content_analysis(original_file_path)
            analysis_results["jobId"] = job_id  # For logging
            
            # 2. Select Dynamic Bitrate Ladder Rungs
            selected_ladder = self.select_ladder_rungs(analysis_results)
            if not selected_ladder:
                raise Exception("No suitable bitrate ladder rungs selected based on analysis.")
                
            # 3. Create Analytics record
            analytics_doc = {
                "jobId": job_id,
                "inputFileSize": analysis_results["inputFileSize"],
                "durationSeconds": analysis_results["durationSeconds"],
                "resolution": analysis_results["resolution"],
                "codec": analysis_results["codec"],
                "frameRate": analysis_results["frameRate"],
                "sceneChangeCount": analysis_results["sceneChangeCount"],
                "opencvMotionScore": float(analysis_results["opencvMotionScore"]),
                "spatialComplexityScore": analysis_results["spatialComplexityScore"],
                "representationMetrics": {},
                "createdAt": datetime.now(),
                "updatedAt": datetime.now()
            }
            
            analytics_result = analytics_collection.insert_one(analytics_doc)
            analytics_id = analytics_result.inserted_id
            
            # Update job with analytics reference
            encoding_jobs_collection.update_one(
                {"jobId": job_id},
                {"$set": {"status": "encoding", "analytics": analytics_id}}
            )
            
            # 4. Encode Representations using Dynamic Ladder
            print(f"[Job {job_id}] Starting encoding for {len(selected_ladder)} representations...")
            
            # Simulate separate encode passes for clarity
            print(f"[Job {job_id}] Note: Simulating separate encode passes for clarity. Combine for efficiency.")
            
            # Clear and recreate output directory
            shutil.rmtree(output_directory, ignore_errors=True)
            os.makedirs(output_directory, exist_ok=True)
            
            # Encode each representation separately
            for rung in selected_ladder:
                rep_id = f"rep_{rung['height']}p"
                rep_dir = os.path.join(output_directory, rep_id)
                os.makedirs(rep_dir, exist_ok=True)
                playlist_path = os.path.join(rep_dir, "playlist.m3u8")  # HLS example
                
                print(f"[Job {job_id}] Encoding {rep_id} (CRF {rung['crf']})...")
                
                encode_args = [
                    FFMPEG_PATH, '-hide_banner', '-loglevel', 'warning', 
                    '-i', original_file_path,
                    '-vf', f"scale=-2:{rung['height']}:flags=lanczos",
                    '-c:v', 'libx264', '-preset', 'slow', 
                    '-crf', str(rung['crf']),
                    '-x264-params', 'keyint=60:min-keyint=60', 
                    '-movflags', '+faststart',
                    '-profile:v', 'high',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '128k',  # Audio
                    # HLS specific args
                    '-f', 'hls',
                    '-hls_time', str(HLS_SEGMENT_DURATION),
                    '-hls_playlist_type', 'vod',
                    '-hls_segment_filename', os.path.join(rep_dir, 'segment%05d.ts'),
                    '-start_number', '0',
                    playlist_path
                ]
                
                encode_start_time = time.time()
                run_command(encode_args, f"{rep_id} encoding")
                encode_duration = time.time() - encode_start_time
                total_encoding_time += encode_duration
                
                # Calculate metrics for this representation
                try:
                    avg_bitrate = self.calculate_average_bitrate(
                        rep_dir, 
                        r'segment.*\.ts$', 
                        analysis_results["durationSeconds"]
                    )
                    
                    # Sample quality metrics at a few points
                    quality_metrics = []
                    sample_points = [
                        30,  # 30 seconds in
                        analysis_results["durationSeconds"] * 0.5,  # Middle
                        min(analysis_results["durationSeconds"] - 30, analysis_results["durationSeconds"] * 0.75)  # Later
                    ]
                    
                    # Updated code block in process_job()
                    for sample_time in sample_points:
                        if sample_time <= 0 or sample_time >= analysis_results["durationSeconds"]:
                            continue
                        
                    # Get segment for comparison
                    segment_file = self._find_segment_at_time(rep_dir, sample_time)
                    if not segment_file:
                        continue
                        
                    # Get original segment with proper parameters
                    original_segment = self.metrics.get_original_segment_path(
                        original_file_path=original_file_path,
                        start_time=sample_time,
                        duration=HLS_SEGMENT_DURATION,
                        job_id=job_id,
                        target_resolution={"width": -2, "height": rung["height"]}
                    )
                            
                    # Calculate VMAF etc.
                    segment_metrics = self.metrics.calculate_metrics(
                        os.path.join(rep_dir, segment_file),
                        original_segment
                    )
                    quality_metrics.append(segment_metrics)
                    
                    # Clean up
                    safe_unlink(original_segment)
                        
                    # Calculate average metrics
                    avg_metrics = {}
                    for metric in ["vmaf", "psnr", "ssim"]:
                        values = [m.get(metric, 0) for m in quality_metrics if metric in m]
                        avg_metrics[metric] = sum(values) / len(values) if values else 0
                        
                    # Store metrics for this representation
                    rep_metrics = {
                        "height": rung["height"],
                        "crf": rung["crf"],
                        "targetBitrateKbps": rung["bitrateKbps"],
                        "actualBitrateKbps": avg_bitrate,
                        "encodingTimeSeconds": encode_duration,
                        "qualityMetrics": avg_metrics
                    }
                    
                    representation_metrics[rep_id] = rep_metrics
                    
                    print(f"[Job {job_id}] {rep_id} stats: {avg_bitrate:.2f} kbps, VMAF: {avg_metrics.get('vmaf', 0):.2f}")
                    
                except Exception as metrics_error:
                    print(f"[Job {job_id}] Error calculating metrics for {rep_id}: {str(metrics_error)}")
                    representation_metrics[rep_id] = {
                        "height": rung["height"],
                        "crf": rung["crf"],
                        "targetBitrateKbps": rung["bitrateKbps"],
                        "encodingTimeSeconds": encode_duration,
                        "error": str(metrics_error)
                    }
            
            # 5. Create master playlist
            master_playlist_path = os.path.join(output_directory, "master.m3u8")
            with open(master_playlist_path, 'w') as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:3\n")
                
                # Add each representation
                for rung in selected_ladder:
                    rep_id = f"rep_{rung['height']}p"
                    bandwidth = representation_metrics.get(rep_id, {}).get("actualBitrateKbps", rung["bitrateKbps"]) * 1000
                    
                    f.write(f"#EXT-X-STREAM-INF:BANDWIDTH={int(bandwidth)},RESOLUTION={rung['height']}p\n")
                    f.write(f"{rep_id}/playlist.m3u8\n")
                    
            # 6. Calculate File Size Reduction
            total_output_size = calculate_dir_size(output_directory)
            output_to_input_ratio = total_output_size / analysis_results["inputFileSize"]
            file_size_reduction_percent = (1 - output_to_input_ratio) * 100
            input_size = analysis_results["inputFileSize"]
            size_reductions = []

            for rung in selected_ladder:
                rep_dir = os.path.join(output_directory, f"rep_{rung['height']}p")
                if not os.path.exists(rep_dir):
                    continue
                    
                rep_size = calculate_dir_size(rep_dir)
                if input_size > 0 and rep_size > 0:
                    reduction = (1 - (rep_size / input_size)) * 100
                    size_reductions.append(reduction)
                    print(f"[Job {job_id}] {rung['height']}p size: {rep_size/1e6:.2f}MB "
                        f"(Input: {input_size/1e6:.2f}MB, Reduction: {reduction:.2f}%)")
            
            file_size_reduction_percent = sum(size_reductions)/len(size_reductions) if size_reductions else 0


            # 7. Update Analytics with results
            analytics_collection.update_one(
                {"_id": analytics_id},
                {"$set": {
                    "representationMetrics": representation_metrics,
                    "avgSizeReductionPerRep": file_size_reduction_percent,
                    "maxSizeReduction": min(size_reductions) if size_reductions else 0,
                    "totalOutputSize": calculate_dir_size(output_directory),
                    "updatedAt": datetime.now()
                }}
            )
            
            # 8. Update job status
            encoding_jobs_collection.update_one(
                {"jobId": job_id},
                {"$set": {
                    "status": "completed",
                    "endTime": datetime.now()
                }}
            )
            
            print(f"[Job {job_id}] Processing completed successfully.")
            print(f"[Job {job_id}] File size reduction: {file_size_reduction_percent:.2f}%")
            
        except Exception as error:
            print(f"[Job {job_id}] Error processing job: {str(error)}")
            try:
                # Update job status to error
                encoding_jobs_collection.update_one(
                    {"jobId": job_id},
                    {"$set": {
                        "status": "error",
                        "errorMessage": str(error),
                        "endTime": datetime.now()
                    }}
                )
                
                # Clean up
                if os.path.exists(output_directory):
                    shutil.rmtree(output_directory, ignore_errors=True)
                    
            except Exception as cleanup_error:
                print(f"[Job {job_id}] Error during cleanup: {str(cleanup_error)}")
    
    def _find_segment_at_time(self, directory: str, time_seconds: float) -> Optional[str]:
        """Find the segment file that contains the specified time"""
        segment_index = int(time_seconds // HLS_SEGMENT_DURATION)

        patterns_to_try = [
            f"segment{segment_index:05d}.ts",
            f"segment{segment_index+1:05d}.ts",  # Handle off-by-one
            f"seg{segment_index}.ts"             # Alternative naming
        ]
        
        for pattern in patterns_to_try:
            for file in os.listdir(directory):
                if file == pattern:
                    return file
                    
        print(f"[Job] Warning: No segment found for time {time_seconds}s in {directory}")
        return None

# --- File System Monitor Class ---
class EncodingServiceFileMonitor(FileSystemEventHandler):
    def __init__(self, encoding_manager: EncodingManager):
        self.encoding_manager = encoding_manager
        self.processing_lock = threading.Lock()
        self.already_processing = set()
        
    def on_created(self, event):
        """Process newly created files in the input directory"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        
        # Only process video files
        if not os.path.basename(file_path).lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
            return
            
        # Avoid processing files that are still being copied
        self._process_file_when_ready(file_path)
        
    def _process_file_when_ready(self, file_path: str):
        """Wait until file is ready (not being modified) before processing"""
        # Check if already being processed
        with self.processing_lock:
            if file_path in self.already_processing:
                return
                
            self.already_processing.add(file_path)
            
        def _wait_and_process():
            try:
                # Wait for file to settle (not being modified)
                last_size = -1
                current_size = os.path.getsize(file_path)
                
                while last_size != current_size:
                    last_size = current_size
                    time.sleep(2)
                    
                    if not os.path.exists(file_path):
                        print(f"[Monitor] File disappeared: {file_path}")
                        return
                        
                    current_size = os.path.getsize(file_path)
                    
                # Process file
                print(f"[Monitor] Adding new file to encoding queue: {file_path}")
                self.encoding_manager.add_job(file_path)
                
            except Exception as e:
                print(f"[Monitor] Error processing new file {file_path}: {str(e)}")
            finally:
                with self.processing_lock:
                    self.already_processing.remove(file_path)
                    
        # Start processing thread
        processing_thread = threading.Thread(target=_wait_and_process)
        processing_thread.daemon = True
        processing_thread.start()

# --- Main Function ---
def main():
    """Main entry point for the encoding service"""
    # Create directories if they don't exist
    os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Create encoding manager and file monitor
    encoding_manager = EncodingManager()
    file_monitor = EncodingServiceFileMonitor(encoding_manager)
    
    # Create file system observer
    observer = Observer()
    observer.schedule(file_monitor, INPUT_DIRECTORY, recursive=False)
    observer.start()
    
    print(f"[Service] Video Encoding Service started.")
    print(f"[Service] Monitoring input directory: {INPUT_DIRECTORY}")
    print(f"[Service] Outputs will be saved to: {OUTPUT_DIRECTORY}")
    print(f"[Service] Max concurrent jobs: {MAX_CONCURRENT_JOBS}")
    
    try:
        # Check for existing files in input directory
        for file_name in os.listdir(INPUT_DIRECTORY):
            file_path = os.path.join(INPUT_DIRECTORY, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                print(f"[Service] Found existing file: {file_name}")
                encoding_manager.add_job(file_path)
                
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("[Service] Shutting down...")
        observer.stop()
        
    observer.join()

if __name__ == "__main__":
    main()               