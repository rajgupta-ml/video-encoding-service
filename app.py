from flask import Flask, request, jsonify, send_from_directory, render_template, url_for, Response
import os
import json
import uuid
import time
import threading
from datetime import datetime
from bson import json_util, ObjectId
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from motion_analyzer import run_content_aware_encoding, save_to_mongodb, DateTimeEncoder
from flask_cors import CORS
import redis
import boto3
import shutil

app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MONGODB_URI'] = os.environ.get('MONGODB_URI')
app.config['DB_NAME'] = 'video_encoding'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv', 'mov', 'webm'}
app.config['S3_BUCKET_NAME'] = os.environ.get('S3_BUCKET_NAME')
app.config['AWS_REGION'] = os.environ.get('AWS_REGION')

# --- Directory Setup ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Redis Connection ---
# Connect to the Redis service named 'redis' in docker-compose.
# decode_responses=True ensures that we get strings back from Redis, not bytes.
redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


def upload_directory_to_s3(directory_path, bucket_name, s3_prefix):
    """Uploads an entire directory to an S3 bucket."""
    s3_client = boto3.client('s3', region_name=app.config['AWS_REGION'])
    print(f"Uploading directory {directory_path} to s3://{bucket_name}/{s3_prefix}")

    for root, _, files in os.walk(directory_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Create a relative path for the S3 key
            relative_path = os.path.relpath(local_path, directory_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            # Set content type for HLS files
            content_type = 'application/octet-stream'
            if filename.endswith('.m3u8'):
                content_type = 'application/x-mpegURL'
            elif filename.endswith('.ts'):
                content_type = 'video/MP2T'

            try:
                s3_client.upload_file(
                    local_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': content_type}
                )
                print(f"  Successfully uploaded {local_path} to {s3_key}")
            except Exception as e:
                print(f"  Failed to upload {local_path}: {e}")
                raise e

def cleanup_local_files(job_id, original_video_path):
    """Deletes the original uploaded video and the output directory."""
    print(f"Cleaning up local files for job {job_id}")
    # Delete the original uploaded file
    try:
        if os.path.exists(original_video_path):
            os.remove(original_video_path)
            print(f"  Deleted original video: {original_video_path}")
    except Exception as e:
        print(f"  Error deleting original video: {e}")

    # Delete the output directory
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"  Deleted output directory: {output_dir}")
    except Exception as e:
        print(f"  Error deleting output directory: {e}")

# --- Utility Functions & JSON Encoding ---
class CustomJSONEncoder(json.JSONEncoder):
    """Custom encoder to handle MongoDB ObjectId and datetime objects."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

def mongo_to_json_serializable(obj):
    """Recursively convert MongoDB documents to be JSON-serializable."""
    if isinstance(obj, dict):
        return {k: mongo_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [mongo_to_json_serializable(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db():
    client = MongoClient(app.config['MONGODB_URI'])
    return client[app.config['DB_NAME']]



def encode_video_async(video_path, job_id):
    """The main background task for video encoding, S3 upload, and cleanup."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Helper functions defined first ---
    def update_job_in_redis(status, progress=None, error=None):
        """Helper function to update job status in Redis."""
        # This function checks if the job hasn't been cancelled before updating.
        job_data_str = redis_client.get(job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            # Stop updating if the job has been cancelled by the user
            if job_data.get('status') == 'cancelled':
                print(f"Job {job_id} was cancelled. Halting progress updates.")
                return

            job_data['status'] = status
            if progress is not None:
                job_data['progress'] = progress
            if error:
                job_data['error'] = error
            
            expiry = 3600 if status in ['failed', 'cancelled'] else 86400
            redis_client.set(job_id, json.dumps(job_data), ex=expiry)

    def progress_callback(percent):
        """Callback to update progress in Redis."""
        update_job_in_redis('processing', progress=percent)
        print(f"Job {job_id} progress: {percent}%")

    # --- Main logic starts here, at the correct indentation level ---
    try:
        start_time = time.time()
        update_job_in_redis('processing', progress=5)

        results = run_content_aware_encoding(video_path, output_dir, progress_callback=progress_callback, job_id=job_id)

        if results is None or 'error' in results:
            raise Exception(results.get('error', 'Encoding failed'))              

        update_job_in_redis('uploading', progress=95)

        bucket_name = app.config['S3_BUCKET_NAME']
        upload_directory_to_s3(output_dir, bucket_name, job_id)
        s3_master_playlist_url = f"https://{bucket_name}.s3.{app.config['AWS_REGION']}.amazonaws.com/{job_id}/{job_id}_master.m3u8"
        representation_s3_urls = {}
        for rep_id in results.get("representationMetrics", {}).keys():
            representation_s3_urls[rep_id] = f"https://{bucket_name}.s3.{app.config['AWS_REGION']}.amazonaws.com/{job_id}/{rep_id.replace('rep', job_id)}/playlist.m3u8"
        print(representation_s3_urls)
        results['s3_master_playlist_url'] = s3_master_playlist_url
        results['representation_s3_urls'] = representation_s3_urls
        results['jobId'] = job_id
        results['originalFilename'] = os.path.basename(video_path)
        print("result", results)
        save_to_mongodb(results, app.config['MONGODB_URI'])
        print(f"Job {job_id} metadata saved to MongoDB.")
        cleanup_local_files(job_id, video_path)
        redis_client.delete(job_id)
    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        update_job_in_redis('failed', error=str(e))
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/health-check")
def healthCheck():
    response_data = {"success": True}
    status_code = 200
    return jsonify(response_data), status_code

@app.route('/api/videos', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or file type'}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(file_path)

    # Initialize job tracking in Redis
    job_data = {
        'id': job_id,
        'originalFilename': filename,
        'status': 'queued',
        'uploadedAt': datetime.now().isoformat(),
        'progress': 0
    }
    # Set an expiry of 24 hours to clean up stuck jobs
    redis_client.set(job_id, json.dumps(job_data), ex=86400)

    # Start encoding in a background thread
    threading.Thread(target=encode_video_async, args=(file_path, job_id)).start()

    return jsonify({
        'jobId': job_id,
        'status': 'queued',
        'message': 'Video uploaded and queued for processing'
    }), 202

@app.route('/api/videos/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    # First, check Redis for active jobs
    job_data_str = redis_client.get(job_id)
    if job_data_str:
        job_info = json.loads(job_data_str)
        return jsonify({
            'jobId': job_id,
            'status': job_info.get('status'),
            'details': job_info
        })

    # If not in Redis, it might be completed. Check the database.
    db = get_db()
    result = db.encoding_jobs.find_one({'jobId': job_id})
    if result:
        details = {
            "progress": 100
        }
        return Response(json_util.dumps({
            'jobId': job_id,
            'status': 'completed',
            'details': details, 
            "progress" : 100,
            'results': result
        }), mimetype='application/json')

    return jsonify({'error': 'Job not found or has expired'}), 404
    

@app.route('/api/videos/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    # Results for completed jobs are only in the database
    db = get_db()
    result = db.encoding_jobs.find_one({'jobId': job_id})
    if result:
        return jsonify(mongo_to_json_serializable(result))

    return jsonify({'error': 'Results not found. The job may still be processing or failed.'}), 404

@app.route('/api/videos/history', methods=['GET'])
def get_encoding_history():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    db = get_db()
    total = db.encoding_jobs.count_documents({})
    results = list(db.encoding_jobs.find({})
                  .sort('completedAt', -1)
                  .skip((page - 1) * limit)
                  .limit(limit))

    return jsonify({
        'total': total,
        'page': page,
        'perPage': limit,
        'results': mongo_to_json_serializable(results)
    })

@app.route('/api/videos/<job_id>/metrics', methods=['GET'])
def get_video_metrics(job_id):
    db = get_db()
    result = db.encoding_jobs.find_one({'jobId': job_id})

    if not result:
        return jsonify({'error': 'Job not found'}), 404
    
    # Extract just the metrics portion
    metrics = {
        'jobId': job_id,
        'originalFilename': result.get('originalFilename', ''),
        'opencvMotionScore': result.get('opencvMotionScore', 0),
        'spatialComplexityScore': result.get('spatialComplexityScore', 0),
        'sceneChangeCount': result.get('sceneChangeCount', 0),
        'avgSizeReductionPerRep': result.get('avgSizeReductionPerRep', 0),
        'representationMetrics': result.get('representationMetrics', {})
    }
    
    return jsonify(metrics)

@app.route('/api/videos/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    # 1. Check if the job is active in Redis
    job_data_str = redis_client.get(job_id)

    if not job_data_str:
        # If not in Redis, check if it's already completed in the database
        db = get_db()
        if db.encoding_jobs.find_one({'jobId': job_id}):
            return jsonify({'error': 'Cannot cancel a completed job'}), 400
        else:
            return jsonify({'error': 'Job not found or has expired'}), 404

    # 2. If the job is in Redis, update its status to 'cancelled'
    try:
        job_data = json.loads(job_data_str)
        job_data['status'] = 'cancelled'

        # Save the updated data back to Redis and set it to expire in an hour
        redis_client.set(job_id, json.dumps(job_data), ex=3600)

        # Note: This marks the job as cancelled. A more advanced implementation
        # would be needed to actually terminate the running ffmpeg process.
        
        return jsonify({'jobId': job_id, 'status': 'cancelled'})

    except json.JSONDecodeError:
        return jsonify({'error': 'Failed to process job data from Redis'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    db = get_db()
    total_videos = db.encoding_jobs.count_documents({})
    
    # Get active/queued jobs from Redis
    active_count = 0
    queued_count = 0
    # Note: KEYS can be slow on large databases. For this app's scale, it's acceptable.
    job_keys = redis_client.keys('*') 
    if job_keys:
        job_values = redis_client.mget(job_keys)
        for job_str in job_values:
            if job_str:
                job = json.loads(job_str)
                if job.get('status') == 'processing':
                    active_count += 1
                elif job.get('status') == 'queued':
                    queued_count += 1

    return jsonify({
        'totalCompletedVideos': total_videos,
        'activeJobs': active_count,
        'queuedJobs': queued_count
    })

@app.route('/api/videos/<job_id>/delete', methods=['DELETE'])
def delete_job(job_id):
    # Delete from database
    db = get_db()
    deleted_in_db = db.encoding_jobs.delete_one({'jobId': job_id}).deleted_count > 0

    # Delete from Redis if it exists
    deleted_in_redis = redis_client.delete(job_id) > 0

    # Delete files from filesystem
    deleted_files = False
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        deleted_files = True
    
    uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(f"{job_id}_")]
    for file in uploaded_files:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        deleted_files = True

    return jsonify({
        'jobId': job_id,
        'deleted': any([deleted_in_db, deleted_in_redis, deleted_files]),
        'message': 'Job and associated files deleted successfully'
    })

@app.route('/player/<job_id>')
def video_player(job_id):
    db = get_db()
    job = db.encoding_jobs.find_one({'jobId': job_id})
    if not job:
        return "Job not found", 404
        
    # Get the S3 URL from the database record
    s3_master_url = job.get('s3_playlist_url')
    if not s3_master_url:
         return "S3 playlist URL not found for this job.", 404
         
    # Pass the absolute S3 URL to the template
    return render_template('player.html', job_id=job_id, master_url=s3_master_url)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)