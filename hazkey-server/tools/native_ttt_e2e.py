# SPDX-License-Identifier: Unlicense
from pathlib import Path
import json,os,socket,struct,subprocess,sys,time,statistics,signal,threading
import pynvml
import argparse,tempfile,shutil
parser=argparse.ArgumentParser(description="Isolated Hazkey native TTT integration/performance probe")
parser.add_argument('--model',required=True,type=Path)
parser.add_argument('--server',type=Path,default=Path.home()/'.local/lib/hazkey/hazkey-server')
parser.add_argument('--library-dir',type=Path,default=Path.home()/'.local/lib/hazkey/libllama')
parser.add_argument('--config',type=Path,default=Path.home()/'.config/hazkey/config.json')
parser.add_argument('--device',default='CUDA1')
parser.add_argument('--nvml-index',type=int,default=1)
parser.add_argument('--work-dir',type=Path)
args=parser.parse_args()
root=args.work_dir or Path(tempfile.mkdtemp(prefix='hazkey-native-e2e-'))
if root.exists() and any(root.iterdir()):parser.error('--work-dir must be empty')
root.mkdir(parents=True,exist_ok=True)
proto=root/'proto';proto.mkdir()
protocol=Path(__file__).resolve().parents[2]/'protocol'
subprocess.run(['protoc','-I',str(protocol),'--python_out='+str(proto),*[str(protocol/name) for name in ['base.proto','commands.proto','config.proto']]],check=True)
sys.path.insert(0,str(proto))
import base_pb2,commands_pb2
root.mkdir(exist_ok=True)
env=os.environ.copy()
for key,name in [('XDG_RUNTIME_DIR','runtime'),('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('XDG_STATE_HOME','state'),('XDG_CACHE_HOME','cache')]:
 d=root/name;d.mkdir(exist_ok=True);d.chmod(0o700);env[key]=str(d)
config_dir=root/'config/hazkey';config_dir.mkdir(exist_ok=True)
profiles=json.loads(args.config.read_text())
for p in profiles:
 p['zenzaiEnable']=True;p['zenzaiBackendDeviceName']=args.device;p['useInputHistory']=False;p['stopStoreNewHistory']=True
(config_dir/'config.json').write_text(json.dumps(profiles))
env['HAZKEY_ZENZAI_MODEL']=str(args.model.resolve())
env['LD_LIBRARY_PATH']=str(args.library_dir.resolve())
# Reproduce the installed dictionary without sharing writable runtime state.
nico=Path.home()/'.local/share/hazkey/dictionaries/dic-nico-intersection-pixiv-google.txt'
if nico.exists():
 dictionaries=root/'data/hazkey/dictionaries';dictionaries.mkdir(parents=True,exist_ok=True)
 shutil.copy2(nico,dictionaries/nico.name)
 cache=root/'cache/hazkey';cache.mkdir(parents=True,exist_ok=True)
 for original in (Path.home()/'.cache/hazkey').glob('nico-regular-merged-v1-*'):
  if original.is_dir() and (original/'complete').is_file():
   shutil.copytree(original,cache/original.name,symlinks=True,copy_function=os.link)
log=open(root/'server.log','wb')
server=subprocess.Popen([str(args.server)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
client=None
results={}
memory_samples=[]
monitor_stop=threading.Event()
pynvml.nvmlInit()
gpu=pynvml.nvmlDeviceGetHandleByIndex(args.nvml_index)
def monitor():
 while not monitor_stop.is_set():
  try:
   for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
    if process.pid==server.pid:memory_samples.append(process.usedGpuMemory/1024**2)
  except pynvml.NVMLError:pass
  monitor_stop.wait(.01)
monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
def read_exact(n):
 value=b''
 while len(value)<n:
  chunk=client.recv(n-len(value))
  if not chunk:raise RuntimeError('server disconnected')
  value+=chunk
 return value
def request(field,**args):
 q=base_pb2.RequestEnvelope();getattr(q,field).SetInParent()
 for k,v in args.items():setattr(getattr(q,field),k,v)
 data=q.SerializeToString();started=time.perf_counter();client.sendall(struct.pack('>I',len(data))+data)
 response=base_pb2.ResponseEnvelope.FromString(read_exact(struct.unpack('>I',read_exact(4))[0]))
 elapsed=(time.perf_counter()-started)*1000
 if response.status!=base_pb2.SUCCESS:raise RuntimeError(response.error_message)
 return response,elapsed
def convert(text):
 request('new_composing_text');request('set_context',context='',anchor=0)
 for c in text:request('input_char',text=c)
 response,elapsed=request('get_candidates',is_suggest=False)
 return [c.text for c in response.candidates.candidates],elapsed
try:
 sock_path=root/f'runtime/hazkey-server.{os.getuid()}.sock'
 for _ in range(200):
  if server.poll() is not None:raise RuntimeError('server exited at startup')
  if sock_path.exists():break
  time.sleep(.05)
 client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);client.settimeout(20);client.connect(str(sock_path))
 baseline=[request('get_current_input_mode')[1] for _ in range(60)]
 before,before_ms=convert('musoukakyou')
 print('BEFORE_CANDIDATES',before,'ms',before_ms,flush=True)
 queue=root/'data/hazkey/zenzai-ttt/word-requests';queue.mkdir(parents=True,exist_ok=True)
 req=queue/'e2e-word.json';req.write_text(json.dumps({'reading':'むそうかきょう','output':'夢想夏郷','max_steps':32},ensure_ascii=False))
 status=req.with_suffix('.status');samples=[];started=time.perf_counter()
 while time.perf_counter()-started<20:
  samples.append(request('get_current_input_mode')[1]);time.sleep(.002)
  if status.exists():
   state=json.loads(status.read_text())
   if state['state'] in ['done','error']:break
 else:raise RuntimeError('word request timed out')
 results['word']=state;results['request_to_done_seconds']=time.perf_counter()-started
 assert state['state']=='done' and state['matched'],state
 after,after_ms=convert('musoukakyou');print('AFTER_CANDIDATES',after,'ms',after_ms,flush=True)
 assert '夢想夏郷' in after,after
 index=after.index('夢想夏郷');request('prefix_complete',index=index)
 progress=root/'data/hazkey/zenzai-ttt/progress.json';start=time.perf_counter();normal_ping=[]
 while time.perf_counter()-start<12:
  normal_ping.append(request('get_current_input_mode')[1]);time.sleep(.005)
  if progress.exists() and json.loads(progress.read_text()).get('processed')==1:break
 else:raise RuntimeError('normal TTT timed out')
 # Measure full conversion requests while a second word is learned and saved.
 for text in ['saikidou','shokumotsuseni']:convert(text)
 conversion_baseline=[convert(['saikidou','shokumotsuseni'][i%2])[1] for i in range(20)]
 cat=queue/'e2e-cat.json';cat.write_text(json.dumps({'reading':'こーでっくすねこ','output':'Codex猫','max_steps':32},ensure_ascii=False))
 conversion_during=[];start_cat=time.perf_counter()
 while time.perf_counter()-start_cat<20:
  conversion_during.append(convert(['saikidou','shokumotsuseni'][len(conversion_during)%2])[1]);time.sleep(.01)
  if cat.with_suffix('.status').exists():
   cat_result=json.loads(cat.with_suffix('.status').read_text())
   if cat_result['state'] in ['done','error']:break
 else:raise RuntimeError('second word request timed out')
 assert cat_result['state']=='done' and cat_result['matched'],cat_result
 results['second_word']=cat_result
 results['normal_progress']=json.loads(progress.read_text());results['normal_reflection_seconds']=time.perf_counter()-start
 def summary(values):
  s=sorted(values);return {'count':len(s),'p50_ms':statistics.median(s),'p95_ms':s[min(len(s)-1,int(len(s)*.95))],'max_ms':max(s)}
 results['full_conversion_baseline']=summary(conversion_baseline);results['full_conversion_during_learning']=summary(conversion_during)
 results['baseline_ping']=summary(baseline);results['word_learning_ping']=summary(samples);results['normal_learning_ping']=summary(normal_ping)
 results['candidates_before']=before;results['candidates_after']=after;results['before_conversion_ms']=before_ms;results['after_conversion_ms']=after_ms
 results['gpu_process_peak_observed_mib']=max(memory_samples) if memory_samples else None
 results['gpu_memory_sampling_interval_ms']=10
 results['nico_dictionary_included']=nico.exists()
 results['isolated_data_root']=str(root);results['passed']=True
 print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
 (root/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
finally:
 monitor_stop.set();monitor_thread.join();pynvml.nvmlShutdown()
 if client:client.close()
 server.terminate()
 try:server.wait(timeout=10)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()
