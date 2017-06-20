from Calibration import AutomaticCalibration
from Utils.DataHandling import reading_and_writing_files as ut

# window length and maximum standard deviation per window to be considered static
static_win_len = 20*50
std_thresh = 100
calibration_type = 6
common_path = 'C:\Users\zwaks\Documents\Workspaces\GitHub\DataScientists\Calibration'

#choose which data to use
#dataFile = common_path + '\data_6pos.csv'
dataFile = common_path + '\data_6pos_ortho.csv'
#dataFile = common_path + '\data_9pos.csv'
#dataFile = common_path + '\data_13pos.csv'

# read data
data = ut.read_export_tool_csv(dataFile)

# Do calculations
cal = AutomaticCalibration(data, static_win_len, std_thresh)
#calib.normalize(1000)
cal.find_window_stds_by_n_samples()
cal.find_static_windows()
cal.calc_static_windows_means()
cal.choose_n_static_windows(calibration_type)

# 6 position calibration******************************
if calibration_type == 6:
    l_ox, l_oy, l_oz, l_sx, l_sy, l_sz = cal.find_calibration_matrix(calibration_type)
    print('Learned parameters')
    print("         Xoffset: ", l_ox, " Xslope: ", l_sx)
    print("         Yoffset: ", l_oy, " Yslope: ", l_sy)
    print("         Zoffset: ", l_oz, " Zslope: ", l_sz)
elif calibration_type == 9:
    l_bx, l_by, l_bz, l_kx, l_ky, l_kz, l_a_yz, l_a_zy, l_a_zx = cal.find_calibration_matrix(calibration_type)
    print("Learned parameters")
    print("         Xoffset: ", l_bx, " Xslope: ", l_kx)
    print("         Yoffset: ", l_by, " Yslope: ", l_ky)
    print("         Zoffset: ", l_bz, " Zslope: ", l_kz)
    print("         Angle yz:", l_a_yz, " Angle zy", l_a_zy, " Angle zx: ", l_a_zx)
else:
    pass

