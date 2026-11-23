import csv
import math
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import rosbag
import rospy
from sensor_msgs.msg import Image, PointCloud

root=pathlib.Path('/home/mxl/catkin_ws')
output=pathlib.Path(tempfile.mkdtemp(prefix='frontend_smoke_',dir=str(root/'output/supergvins')))
config=(root/'src/SuperGVINS/config/visensor_f9p/visensor_left_f9p_config.yaml').read_text()
config=config.replace('/home/mxl/catkin_ws/output/supergvins/sports_field/baseline/',str(output)+'/')
config=config.replace('inference_device: "cpu"','inference_device: "'+os.environ.get('SUPERGVINS_SMOKE_DEVICE','cpu')+'"')
config_path=output/'config.yaml';config_path.write_text(config)
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
env=os.environ.copy();env['ROS_MASTER_URI']=f'http://127.0.0.1:{port}';env['ROS_LOG_DIR']=str(output/'ros_log')
log=(output/'roscore.log').open('w'); node_log=(output/'node.log').open('w')
master=subprocess.Popen(['roscore','-p',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
node=None
try:
    os.environ['ROS_MASTER_URI']=env['ROS_MASTER_URI']
    for _ in range(100):
        try:
            import rosgraph
            if rosgraph.Master('/supergvins_smoke').getPid(): break
        except Exception: time.sleep(.1)
    else: raise RuntimeError('roscore did not start')
    node=subprocess.Popen([str(root/'devel/lib/supergvins_feature_tracker/supergvins_feature_tracker'),
        '_config_file:='+str(config_path),'_gvins_folder:='+str(root/'src/SuperGVINS')+'/'],env=env,stdout=node_log,stderr=subprocess.STDOUT)
    rospy.init_node('supergvins_frontend_smoke',anonymous=True,disable_signals=True)
    received=[]
    sub=rospy.Subscriber('/supergvins_feature_tracker/feature',PointCloud,lambda msg:received.append(msg),queue_size=20)
    pub=rospy.Publisher('/cam1/image_raw',Image,queue_size=20)
    for _ in range(200):
        if node.poll() is not None: raise RuntimeError('frontend exited before subscription')
        if pub.get_num_connections(): break
        time.sleep(.1)
    else: raise RuntimeError('frontend did not subscribe')
    time.sleep(0.5)
    sent=[]
    with rosbag.Bag(str(root/'data/gvins/sports_field.bag')) as bag:
        for _,msg,_ in bag.read_messages(topics=['/cam1/image_raw']):
            pub.publish(msg);sent.append(msg.header.stamp.to_nsec())
            if len(sent)==5: break
            time.sleep(.2)
    deadline=time.time()+45
    while len(received)<len(sent) and time.time()<deadline:
        if node.poll() is not None: raise RuntimeError('frontend exited during processing')
        time.sleep(.1)
    assert len(received)==len(sent),f'received {len(received)} / {len(sent)} frames'
    assert [m.header.stamp.to_nsec() for m in received]==sent,'timestamp mismatch'
    for m in received:
        assert len(m.channels)==5 and all(len(c.values)==len(m.points) for c in m.channels)
        assert len(m.points)<=200
        ids=[int(v) for v in m.channels[0].values]
        assert len(ids)==len(set(ids))
        assert all(p.z == 1 and math.isfinite(p.x) and math.isfinite(p.y) for p in m.points)
        assert all(math.isfinite(v) for c in m.channels for v in c.values)
    checked_velocities=0
    for before,after in zip(received[1:],received[2:]):
        previous={int(before.channels[0].values[i]):p for i,p in enumerate(before.points)}
        dt=(after.header.stamp.to_nsec()-before.header.stamp.to_nsec())/1e9
        for i,p in enumerate(after.points):
            fid=int(after.channels[0].values[i])
            if fid in previous:
                assert math.isclose(after.channels[3].values[i], (p.x-previous[fid].x)/dt, rel_tol=1e-3, abs_tol=1e-3)
                assert math.isclose(after.channels[4].values[i], (p.y-previous[fid].y)/dt, rel_tol=1e-3, abs_tol=1e-3)
                checked_velocities+=1
    assert checked_velocities>0,'no persistent tracks checked'
    with (output/'frontend_frames.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==len(sent)
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
    assert any(int(r['matches_before_ransac'])>0 for r in rows), 'no matches'
    assert any(len(m.points)>0 for m in received), 'no mature features'
    print('SMOKE PASS: sent=',len(sent),'published=',len(received),'point_counts=',[len(m.points) for m in received],'velocities_checked=',checked_velocities)
    print('raw_points=',[r['raw_keypoints'] for r in rows],
          'candidates=',[r['kept_keypoints'] for r in rows],
          'prev_candidates=',[r['prev_candidate_count'] for r in rows],
          'matches=',[r['matches_before_ransac'] for r in rows],
          'ransac_inliers=',[r['ransac_inliers_before_cap'] for r in rows],
          'retained=',[r['retained_tracks'] for r in rows])
    print('artifacts:',output)
finally:
    if node is not None:
        node.terminate()
        try: node.wait(timeout=8)
        except subprocess.TimeoutExpired: node.kill();node.wait()
    master.terminate()
    try: master.wait(timeout=8)
    except subprocess.TimeoutExpired: master.kill();master.wait()
    log.close();node_log.close()
    print('logs:',output)
