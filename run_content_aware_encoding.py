#!/usr/bin/env python
import sys
import os
import json
import time
from motion_analyzer_copy import run_content_aware_encoding, save_to_mongodb, DateTimeEncoder

def main():
    if len(sys.argv) < 2:
        video_file = input("Enter path to video file: ").strip()
    else:
        video_file = sys.argv[1]

    # Set output directory (default or from command line)
    output_dir = "output"
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]

    # Set MongoDB connection string (default or from command line)
    mongo_uri = "mongodb://localhost:27017/"
    if len(sys.argv) >= 4:
        mongo_uri = sys.argv[3]

    if not os.path.isfile(video_file):
        print(f"Error: File does not exist — {video_file}", file=sys.stderr)
        sys.exit(1)

    print("--- Starting Content-Aware Video Encoding with HLS Output ---")
    print(f"Processing file: {video_file}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    results = run_content_aware_encoding(video_file, output_dir)
    total_time = time.time() - start_time
    
    # Add processing time to results
    results["processingTimeSeconds"] = total_time
    
    # Save results to JSON file - use custom encoder for datetime objects
    output_json = f"{output_dir}/{os.path.splitext(os.path.basename(video_file))[0]}_encoding_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, cls=DateTimeEncoder, indent=2)
    
    # Save to MongoDB
    db_id = save_to_mongodb(results, mongo_uri)
    
    # Print summary
    print("\n--- Encoding Results ---")
    print(f"Job ID: {results.get('jobId')}")
    print(f"Motion Score: {results.get('opencvMotionScore', 0):.4f}")
    print(f"Spatial Complexity: {results.get('spatialComplexityScore', 0):.4f}")
    print(f"Scene Changes: {results.get('sceneChangeCount', 0)}")
    print(f"Size Reduction: {results.get('avgSizeReductionPerRep', 0):.2f}%")
    print("\nQuality Metrics:")
    
    for rep_name, rep_data in results.get("representationMetrics", {}).items():
        print(f"\n{rep_name}:")
        print(f"  Resolution: {rep_data.get('height')}p")
        print(f"  CRF: {rep_data.get('crf')}")
        print(f"  Target Bitrate: {rep_data.get('targetBitrateKbps'):.0f} Kbps")
        print(f"  Actual Bitrate: {rep_data.get('actualBitrateKbps', 0):.0f} Kbps")
        print(f"  VMAF: {rep_data.get('qualityMetrics', {}).get('vmaf', 0):.2f}")
        print(f"  PSNR: {rep_data.get('qualityMetrics', {}).get('psnr', 0):.2f}")
        print(f"  SSIM: {rep_data.get('qualityMetrics', {}).get('ssim', 0):.4f}")
    
    print(f"\nHLS files saved to: {output_dir}/{results.get('jobId')}_*")
    print(f"Full results saved to JSON: {output_json}")
    if db_id:
        print(f"Results stored in MongoDB with ID: {db_id}")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
