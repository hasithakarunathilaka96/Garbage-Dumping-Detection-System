import cv2
import time
import datetime
import asone
from asone import ASOne
import sys
import argparse

import socket
import pickle
import os
import ftplib

from loguru import logger
import enum

# Calculate the center coordinates of bounding box
def bb_center(bbox, i):
    return ((bbox[i][2] + bbox[i][0]) / 2, (bbox[i][3] + bbox[i][1]) / 2)

# Calculate the square distance between two center points
def sq_dist(c1, c2):
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

# Function to send recorded evidence video to backend via FTP
def send_video(video_path):

    filename = os.path.basename(video_path)

	# Define FTP connection parameters
    FTP_HOST = '192.168.0.207'
    FTP_USER = 'intellisys'
    FTP_PASS = 'intellisys'
    FTP_DIR = 'Demo/Recorded_Videos'

	# Connect to the FTP server
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    ftp.cwd(FTP_DIR)

	# Upload the video to Backend
    try:
        with open(video_path, 'rb') as file:
            retcode = ftp.storbinary(f'STOR {filename}', file, blocksize=1024*1024)

        if retcode.startswith('226'):
            logger.success(f'Video upload successful: {filename}')
            os.remove(video_path)
        else:
            logger.error(f'Video upload failed: {filename}')
    
    except:
        logger.error(f'Video upload error: {filename}')
    
    finally:
        ftp.quit()

# Function to send the detected object details to the backend
def send_message(new_obj_detect, timestamp):
    host = '192.168.0.207'
    port = 8505

    message = f'{new_obj_detect} was detected @ {timestamp}'

    try:
        mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mysocket.connect((host, port))
        mysocket.send(message.encode('utf-8'))
        mysocket.close()

        logger.success(f'Sent object details: "{message}"')
    
    except:
        logger.error(f'Error sending message: "{message}"')

# Function to record video using saved frames in frame_buffer
def record_video(frame_buffer,new_obj_detect):
	now = datetime.datetime.now()
	timestamp = f'{str(now.date())}_{str(now.hour)}{str(now.minute)}{str(now.second)}'
	path = f'{timestamp}.mp4'
	logger.info(f'Recording the evidence video {path}')
	output = cv2.VideoWriter(path, fourcc, 12,
							(width, height))
	for frame in frame_buffer:
		output.write(frame)
		k = cv2.waitKey(24)
		if k == ord("q"):
			break
	output.release()
	send_video(path)
	send_message(new_obj_detect, timestamp)
	

if __name__ == '__main__':

	logger.info(f'Starting the Garbage Dumping detection System by CS2')

	class_names = ['Bicycle', 'Chair', 'Box', 'Table', 'Plastic bag', 'Flowerpot', 'Luggage and bags', 'Umbrella', 'Shopping trolley', 'Person']
	stream_url = 'rtsp://cs2projs:cs2projs@192.168.0.166/stream2'
	cap = cv2.VideoCapture(stream_url)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fps = cap.get(cv2.CAP_PROP_FPS)
	path = ""
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
	server_ip = '192.168.0.207'
	server_port = 8500

	use_cuda = False
	runstatus = False

	while(True):
		starting_status = False
		recording_status = False
		buffer_size = 10
		frame_buffer = []
		start_time = time.time()

		current_obj_ids = []
		new_obj_detect = ''
		new_obj_id = ''
		person_id = ''

		is_stopping = False
		upload_status = False
		video_discard = False
		
		detect = ASOne(tracker=asone.NORFAIR, detector=asone.YOLOV8N_PYTORCH, weights='yolov8_hdb.pt', use_cuda = use_cuda )
		track = detect.track_video('rtsp://cs2projs:cs2projs@192.168.0.166/stream2', class_names=class_names)
		for bbox_details, frame_details in track:
			bbox_xyxy, ids, scores, class_ids = bbox_details
			frame, frame_num, fps = frame_details
			class_list = [class_names[int(i)] for i in class_ids]

			cv2.imshow('Live Video', frame)

			if(recording_status == False):
				has_person = class_list.count('Person') > 0
				for i, obj in enumerate(class_list):
					is_new_object = obj != 'Person' and ids[i] not in current_obj_ids
				
					if has_person and is_new_object:
						new_obj_detect = obj
						new_obj_id = ids[i]
						current_obj_ids.append(ids[i])
						obj_pos = bb_center(bbox_xyxy, i)
						person_dist = -1

						for j, person in enumerate(class_list):	
							if person == 'Person':
								person_pos = bb_center(bbox_xyxy, j)
								new_person_dist = sq_dist(obj_pos, person_pos)
								
								if new_person_dist < person_dist or person_dist == -1:
									person_dist = new_person_dist
									person_id = ids[j]

						logger.info(f'Person id: {person_id}, {new_obj_detect} id: {new_obj_id} detected.')
						recording_status = True
						break
					if (recording_status==True):
						frame_buffer.append(frame)
						break

			# Saving frsmes
			if (recording_status == True):
				frame_buffer.append(frame)

			# Send frames for Live streaming
			ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 8])
			x_as_bytes = pickle.dumps(buffer)
			mysocket.sendto((x_as_bytes), (server_ip, server_port))

			if (recording_status == True):
				if (person_id not in ids) and (is_stopping == False):
					logger.debug(f'Person id: {person_id} left, waiting for 5 seconds.')
					time_to_stop = time.time() + 7
					is_stopping = True

				if (is_stopping == True):
					if person_id in ids:
						logger.debug(f'Person id: {person_id} redetected, resuming recording.')
						is_stopping = False

					elif time.time() > time_to_stop:
						if new_obj_id in ids:
							upload_status = True
						else:
							logger.debug(f'Object id: {new_obj_id} not found, Video Discarding.')
							video_discard= True
						is_stopping = False


			# Recording and Uploading video
			if(upload_status == True and recording_status == True):
				record_video(frame_buffer,new_obj_detect)
				recording_status = False
				upload_status = False
				frame_buffer = []
			
			if(video_discard == True):
				frame_buffer = []
				recording_status = False
				video_discard = False

			if (class_list.count('Person')==0):
				for i, obj in enumerate(class_list):
					if obj != 'Person' and ids[i] not in current_obj_ids:
						current_obj_ids.append(ids[i])
							
		cap.release()
		cv2.destroyAllWindows()