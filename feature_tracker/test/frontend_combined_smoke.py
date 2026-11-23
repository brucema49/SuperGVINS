import csv,os,pathlib,socket,subprocess,tempfile,time
import rosbag,rospy
root=pathlib.Path('/home/mxl/catkin_ws')
out=pathlib.Path(tempfile.mkdtemp(prefix='combined_smoke_',dir=str(root/'output/supergvins')))
conf=(root/'src/SuperGVINS/config/visensor_f9p/visensor_left_f9p_config.yaml').read_text().replace('/home/mxl/catkin_ws/output/supergvins/sports_field/baseline/',str(out)+'/')
config=out/'config.yaml';config.write_text(conf)
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
env=os.environ.copy();env['ROS_MASTER_URI']=f'http://127.0.0.1:{port}';env['ROS_LOG_DIR']=str(out/'ros_log');os.environ['ROS_MASTER_URI']=env['ROS_MASTER_URI']
logs=[];processes=[]
def start(args,name):
 f=(out/name).open('w');logs.append(f);p=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT);processes.append(p);return p
try:
 core=start(['roscore','-p',str(port)],'roscore.log')
 import rosgraph
 for _ in range(100):
  try:
   if rosgraph.Master('/combined_smoke').getPid():break
  except Exception:time.sleep(.1)
 else:raise RuntimeError('roscore timeout')
 subprocess.check_call(['rosparam','set','/use_sim_time','true'],env=env)
 launch=start(['roslaunch','supergvins','visensor_f9p.launch','config_path:='+str(config)],'launch.log')
 master=rosgraph.Master('/combined_smoke')
 for _ in range(150):
  if launch.poll() is not None:raise RuntimeError('launch exited')
  state=master.getSystemState()
  subs={topic for topic, nodes in state[1]}
  if '/cam1/image_raw' in subs and '/imu0' in subs and '/supergvins_feature_tracker/feature' in subs:break
  time.sleep(.1)
 else:raise RuntimeError('subscribers not ready')
 with rosbag.Bag(str(root/'data/gvins/sports_field.bag')) as bag:
  end=rospy.Time.from_sec(bag.get_start_time()+2.0)
  expected=[msg.header.stamp.to_nsec() for _,msg,_ in bag.read_messages(topics=['/cam1/image_raw'],end_time=end)]
 play=start(['rosbag','play','--clock','--delay','2','-r','0.1','-u','2',str(root/'data/gvins/sports_field.bag')],'play.log')
 if play.wait(timeout=40)!=0:raise RuntimeError('rosbag playback failed')
 deadline=time.time()+55
 rows=[]
 while time.time()<deadline:
  if launch.poll() is not None:raise RuntimeError('launch exited during playback')
  with (out/'frontend_frames.csv').open() as f:rows=list(csv.DictReader(f))
  if len(rows)>=len(expected):break
  time.sleep(.5)
 assert [int(r['stamp_ns']) for r in rows]==expected,f'image coverage {len(rows)} / {len(expected)}'
 assert all(int(r['superpoint_calls'])==1 and
            int(r['kept_keypoints'])<=512 and
            int(r['prev_track_count'])<=200 and
            int(r['prev_candidate_count'])<=512 and
            int(r['matches_before_ransac'])<=512 and
            int(r['matches_before_ransac'])<=min(int(r['prev_candidate_count']),int(r['kept_keypoints'])) and
            int(r['ransac_inliers_before_cap'])<=int(r['matches_before_ransac']) and
            int(r['matches_after_ransac'])==min(int(r['ransac_inliers_before_cap']),200) and
            int(r['retained_tracks'])<=200 and
            int(r['published_points'])<=200 for r in rows)
 assert int(rows[0]['prev_candidate_count'])==0 and int(rows[0]['prev_track_count'])==0
 assert all(int(current['prev_candidate_count'])==int(previous['kept_keypoints']) and
            int(current['prev_track_count'])==int(previous['retained_tracks'])
            for previous,current in zip(rows,rows[1:])), 'previous-frame cache/track counts do not line up'
 print('COMBINED SMOKE PASS frames=',len(rows),'first=',rows[0]['stamp_ns'],'last=',rows[-1]['stamp_ns'])
 print('artifacts:',out)
finally:
 for p in reversed(processes):
  if p.poll() is None:p.terminate()
 for p in reversed(processes):
  try:p.wait(timeout=8)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 for f in logs:f.close()
 print('logs:',out)
