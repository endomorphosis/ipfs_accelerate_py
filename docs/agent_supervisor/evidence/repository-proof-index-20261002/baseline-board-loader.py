import importlib.abc,importlib.util,sys
class BaselineBoard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname == "ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane":
   return importlib.util.spec_from_file_location(fullname,"/tmp/rpi-board-baseline-20261002/baseline_board.py")
sys.meta_path.insert(0,BaselineBoard())
