# Enhanced motion_analyzer.py with HLS support and MongoDB storage - with datetime JSON fix

import cv2
import numpy as np
import subprocess
import json
import os
import time
from datetime import datetime
import uuid
from pymongo import MongoClient
import re

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def analyze_video_motion(video_path):
    """Analyze motion in video and return motion score"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return -1
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize variables for motion analysis
        prev_frame = None
        motion_scores = []
        scene_changes = 0
        
        # Sample frames (analyze every 5th frame to improve speed)
        sample_rate = 5
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert to grayscale for motion analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    frame_diff = cv2.absdiff(prev_frame, gray)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Calculate motion score (percentage of pixels that changed)
                    motion_score = np.count_nonzero(thresh) / thresh.size
                    motion_scores.append(motion_score)
                    
                    # Detect scene changes
                    if motion_score > 0.2:  # Threshold for scene change
                        scene_changes += 1
                
                prev_frame = gray
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall motion score
        if motion_scores:
            avg_motion = sum(motion_scores) / len(motion_scores)
            return avg_motion, scene_changes
        else:
            return 0, 0
    except Exception as e:
        print(f"Error in motion analysis: {e}")
        return -1, 0

def calculate_spatial_complexity(video_path, sample_rate=30):
    """Calculate spatial complexity of the video based on edge detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0.5  # Default medium complexity
            
        complexity_scores = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply Sobel edge detection
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                # Calculate magnitude
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # Normalize and calculate complexity score (0-1)
                complexity = np.mean(magnitude) / 255.0
                complexity_scores.append(complexity)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall complexity score
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            # Normalize to 0-1 range with more practical distribution
            normalized = min(1.0, max(0.0, avg_complexity * 2))
            return normalized
        else:
            return 0.5  # Default medium complexity
    except Exception as e:
        print(f"Error calculating spatial complexity: {e}")
        return 0.5  # Default medium complexity


def create_master_playlist(job_id, output_dir, encoded_representations):
    """Create a master HLS playlist that references all quality variants"""
    master_playlist_path = f"{output_dir}/{job_id}_master.m3u8"
    
    with open(master_playlist_path, 'w') as f:
        # Write the HLS version header
        f.write("#EXTM3U\n")
        f.write("#EXT-X-VERSION:3\n")
        
        # Add each representation to the master playlist
        for res_name, metrics in encoded_representations.items():
            # Extract the resolution (like "360p", "720p") from key names like "rep_360p"
            res = res_name.replace("rep_", "")
            height = metrics["height"]
            bandwidth = int(metrics["actualBitrateKbps"] * 1000)  # Convert to bps
            playlist_relative_path = f"{job_id}_{res}/playlist.m3u8"
            metrics["playlistUrl"] = playlist_relative_path
            # Write the stream info
            f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={metrics.get("width", "?")}x{height}\n')
            
            # Write the playlist path (relative to the master playlist)
            f.write(f'{job_id}_{res}/playlist.m3u8\n')
    
    return master_playlist_path


def suggest_encoding_params(motion_score, spatial_complexity, resolution_height):
    """Suggest encoding parameters based on motion score and resolution"""
    # Blend motion and complexity for content-aware decision making
    content_complexity = (motion_score * 0.7) + (spatial_complexity * 0.3)
    
    # Base CRF values adjusted for content type
    if content_complexity < 0.05:  # Very static simple content
        base_crf = 26
    elif content_complexity < 0.1:  # Low complexity
        base_crf = 24
    elif content_complexity < 0.2:  # Moderate complexity
        base_crf = 22
    else:  # High complexity
        base_crf = 20
    
    # Resolution-specific adjustments
    encoding_params = {}
    
    # 360p
    encoding_params["360p"] = {
        "height": 360,
        "crf": base_crf,
        "targetBitrateKbps": int(400 + 1200 * content_complexity)
    }
    
    # 480p
    encoding_params["480p"] = {
        "height": 480,
        "crf": base_crf - 1,
        "targetBitrateKbps": int(600 + 1800 * content_complexity)
    }
    
    # 720p
    encoding_params["720p"] = {
        "height": 720,
        "crf": base_crf - 1,
        "targetBitrateKbps": int(1000 + 3500 * content_complexity)
    }
    
    # 1080p
    encoding_params["1080p"] = {
        "height": 1080,
        "crf": base_crf - 2,
        "targetBitrateKbps": int(2000 + 5000 * content_complexity)
    }
    
    return encoding_params

def calculate_quality_metrics(source_video, encoded_video):
    """Calculate VMAF, PSNR and SSIM metrics properly"""
    metrics = {"vmaf": 0, "psnr": 0, "ssim": 0}
    
    try:
        # Get source video dimensions
        cap = cv2.VideoCapture(source_video)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create temp directory for logs if it doesn't exist
        os.makedirs("temp_metrics", exist_ok=True)
        
        # VMAF calculation
        vmaf_log = "temp_metrics/vmaf_log.json"
        vmaf_cmd = [
            "ffmpeg", "-i", encoded_video, "-i", source_video,
            "-filter_complex", f"[0:v]scale={orig_width}:{orig_height}[scaled];[scaled][1:v]libvmaf=log_fmt=json:log_path={vmaf_log}:model=version=vmaf_v0.6.1",
            "-f", "null", "-"
        ]
        subprocess.run(vmaf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Parse VMAF results
        if os.path.exists(vmaf_log):
            with open(vmaf_log, "r") as f:
                vmaf_data = json.load(f)
                metrics["vmaf"] = round(vmaf_data.get("pooled_metrics", {}).get("vmaf", {}).get("mean", 0), 2)
        
        # PSNR calculation - direct filter output
        psnr_log = "temp_metrics/psnr_log.txt"
        psnr_cmd = [
            "ffmpeg", "-i", encoded_video, "-i", source_video,
            "-filter_complex", f"[0:v]scale={orig_width}:{orig_height}[scaled];[scaled][1:v]psnr=stats_file={psnr_log}",
            "-f", "null", "-"
        ]
        psnr_result = subprocess.run(psnr_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Extract PSNR from stderr output (more reliable than the log file)
        stderr_output = psnr_result.stderr
        for line in stderr_output.split('\n'):
            if "average" in line and "psnr" in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        psnr_value = float(parts[1].strip().split()[0])
                        metrics["psnr"] = round(psnr_value, 2)
                    except (ValueError, IndexError):
                        pass
        
        # SSIM calculation - direct filter output
        ssim_log = "temp_metrics/ssim_log.txt"
        ssim_cmd = [
            "ffmpeg", "-i", encoded_video, "-i", source_video,
            "-filter_complex", f"[0:v]scale={orig_width}:{orig_height}[scaled];[scaled][1:v]ssim=stats_file={ssim_log}",
            "-f", "null", "-"
        ]
        ssim_result = subprocess.run(ssim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Extract SSIM from stderr output
        stderr_output = ssim_result.stderr
        for line in stderr_output.split('\n'):
            if "All:" in line and "ssim" in line:
                parts = line.split('All:')
                if len(parts) >= 2:
                    try:
                        ssim_value = float(parts[1].strip().split()[0])
                        metrics["ssim"] = round(ssim_value, 4)
                    except (ValueError, IndexError):
                        pass
    
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
    
    return metrics

def encode_video_to_hls(input_path, output_dir, params, job_id, resolution_name):
    """Encode video to HLS format with specified parameters"""
    height = params["height"]
    crf = params["crf"]
    target_bitrate = params["targetBitrateKbps"]
    
    # Calculate width maintaining aspect ratio
    cap = cv2.VideoCapture(input_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    width = int((orig_width / orig_height) * height)
    width = width - (width % 2)  # Ensure even width
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First create an intermediate MP4 file for quality metrics calculation
    temp_mp4 = f"{output_dir}/temp_{resolution_name}.mp4"
    
    # Basic FFmpeg command for H.264 encoding with CRF
    ffmpeg_mp4_cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", 
        "-crf", str(crf),
        "-preset", "medium",
        "-b:v", f"{target_bitrate}k",
        "-maxrate", f"{int(target_bitrate * 1.5)}k",
        "-bufsize", f"{target_bitrate * 2}k",
        "-vf", f"scale={width}:{height}",
        "-y", temp_mp4
    ]
    
    # Run MP4 encoding and time it
    start_time = time.time()
    subprocess.run(ffmpeg_mp4_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoding_time = time.time() - start_time
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(input_path, temp_mp4)
    
    # Now create HLS files
    hls_output_dir = f"{output_dir}/{job_id}_{resolution_name}"
    os.makedirs(hls_output_dir, exist_ok=True)
    
    hls_cmd = [
        "ffmpeg", "-i", temp_mp4,
        "-c:v", "copy",  # No re-encoding needed since we already encoded to MP4
        "-start_number", "0",
        "-hls_time", "10",  # 10-second segments
        "-hls_list_size", "0",  # Keep all segments in playlist
        "-f", "hls",
        f"{hls_output_dir}/playlist.m3u8"
    ]
    
    # Run HLS creation
    subprocess.run(hls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Calculate actual bitrate from MP4 file
    file_size_bytes = os.path.getsize(temp_mp4)
    duration_sec = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", temp_mp4],
        stderr=subprocess.PIPE
    ).decode().strip())
    
    actual_bitrate_kbps = (file_size_bytes * 8) / (duration_sec * 1000)
    
    # Return metrics
    return {
        "height": height,
        "width" : width,
        "crf": crf,
        "targetBitrateKbps": target_bitrate,
        "actualBitrateKbps": actual_bitrate_kbps,
        "encodingTimeSeconds": encoding_time,
        "qualityMetrics": quality_metrics
    }

def save_to_mongodb(results, mongo_uri="mongodb://localhost:27017/", db_name="video_encoding", collection_name="encoding_jobs"):
    """Save encoding results to MongoDB"""
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Insert results and get the inserted ID
        result = collection.insert_one(results)
        print(f"Results saved to MongoDB with ID: {result.inserted_id}")
        
        # Close the connection
        client.close()
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None

def run_content_aware_encoding(video_path, output_dir="output", progress_callback=None, job_id=None):

    try : 
        """Full content-aware encoding pipeline with HLS output and progress callback support"""
        # Generate a unique job ID
        if job_id is None:
            job_id = str(uuid.uuid4())[:12]

        # Start progress at 0
        if progress_callback:
            progress_callback(0)
        
        # Step 1: Analyze motion and scene changes
        motion_score, scene_changes = analyze_video_motion(video_path)
        if motion_score < 0:
            return {"error": "Failed to analyze motion"}
        if progress_callback:
            progress_callback(10)
        
        # Step 2: Calculate spatial complexity
        spatial_complexity = calculate_spatial_complexity(video_path)
        if progress_callback:
            progress_callback(20)
        
        # Step 3: Extract basic video properties
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_file_size = os.path.getsize(video_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        codec = "h264"  # Assumed codec for simplicity
        cap.release()
        if progress_callback:
            progress_callback(30)
        
        # Step 4: Suggest encoding parameters
        encoding_params = suggest_encoding_params(motion_score, spatial_complexity, height)

        
        # Step 5: Initialize results dictionary
        results = {
            "jobId": job_id,
            "inputFileSize": input_file_size,
            "durationSeconds": duration,
            "resolution": {"width": width, "height": height},
            "codec": codec,
            "frameRate": fps,
            "sceneChangeCount": scene_changes,
            "opencvMotionScore": motion_score,
            "spatialComplexityScore": spatial_complexity,
            "representationMetrics": {},
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        total_output_size = 0
        size_reductions = []
        
        # Step 6: Encode each resolution and update progress accordingly
        reps = list(encoding_params.items())
        reps_count = len(reps)
        
        # We will divide 50% of progress across all representations
        # Because we already used 0–30% till now, rest (30–90%) is for encoding
        for idx, (res, params) in enumerate(reps):
            print(f"Encoding {res} representation...")
            
            rep_metrics = encode_video_to_hls(video_path, output_dir, params, job_id, res)
            results["representationMetrics"][f"rep_{res}"] = rep_metrics
            
            # Calculate size for this representation
            temp_mp4 = f"{output_dir}/temp_{res}.mp4"
            if os.path.exists(temp_mp4):
                rep_size = os.path.getsize(temp_mp4)
                total_output_size += rep_size
                
                size_reduction = 100 * (1 - (rep_size / input_file_size))
                size_reductions.append(size_reduction)
            
            # Update progress based on completed encodings
            if progress_callback:
                encoding_progress = 30 + int(((idx + 1) / reps_count) * 50)  # 30-80%
                progress_callback(encoding_progress)
        
        # Step 7: After encoding all representations, update size reduction
        if size_reductions:
            results["avgSizeReductionPerRep"] = sum(size_reductions) / len(size_reductions)
            results["maxSizeReduction"] = max(size_reductions)
        else:
            results["avgSizeReductionPerRep"] = 0
            results["maxSizeReduction"] = 0
        
        results["totalOutputSize"] = total_output_size
        results["updatedAt"] = datetime.utcnow()
        
        # Step 8: Final progress before saving
        if progress_callback:
            progress_callback(90)
            
        # Create master playlist that references all representations
        
        master_playlist = create_master_playlist(job_id, output_dir, results["representationMetrics"])
        results["masterPlaylist"] = master_playlist
        
        # Here you would usually save to MongoDB or perform final steps
        # Assume MongoDB saving in encode_video_async after this
        
        # Step 9: Mark 100% after everything is done
        if progress_callback:
            progress_callback(100)
        
        return results
    except Exception as e : 
        print(f"Error in run_content_aware_encoding: {e}")
        if progress_callback:
            progress_callback(0)  # Reset progress on error
        return {"error": str(e)}