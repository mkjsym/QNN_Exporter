"""Apply the established mixed W4/W8 A16 recipe with unchanged train calibration."""
import sys
import vlm_data_scale_export_common as common
sys.modules['vlm_residual8_export_common']=common
from export_vlm_residual8_draft import main

if __name__=='__main__':main()
