"""Run the existing B8/open7 static graph audit on the selected checkpoint."""
import sys
import vlm_data_scale_export_common as common
sys.modules['vlm_residual8_export_common']=common
from check_vlm_residual8_export import main

if __name__=='__main__':main()
