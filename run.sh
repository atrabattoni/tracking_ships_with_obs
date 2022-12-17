mkdir -p data
mkdir -p figs
cd 0_data
echo "======================= 0_data / 1_station.py =========================="
python 1_station.py
echo "======================= 0_data / 2_tracks.py ==========================="
python 2_tracks.py
echo "======================= 0_data / 3_streams.py =========================="
python 3_streams.py
cd ..
cd "1_exp&obs"
echo "======================= 1_exp&obs / 1_spectrogram.py ==================="
python 1_spectrogram.py
echo "======================= 1_exp&obs / 2_ais.py ==========================="
python 2_ais.py
cd ..
cd 2_method
echo "======================= 2_method / 1_adjust_model.py ==================="
python 1_adjust_model.py
echo "======================= 2_method / 2_data_model.py ====================="
python 2_data_model.py
echo "======================= 2_method / 3_branch_association.py ============="
python 3_branch_association.py
echo "======================= 2_method / 3_direction.py ======================"
python 3_direction.py
echo "======================= 2_method / 3_distance.py ======================="
python 3_distance.py
cd ..
cd 3_detection
echo "======================= 3_detection / 1_direction.py ==================="
python 1_direction.py
echo "======================= 3_detection / 1_distance.py ===================="
python 1_distance.py
echo "======================= 3_detection / 2_error.py ======================="
python 2_error.py
cd ..
cd 4_tracking 
echo "======================= 4_tracking / 1_segments.py ====================="
python 1_segments.py
echo "======================= 4_tracking / 2_process.py ======================"
python 2_process.py
echo "======================= 4_tracking / 3_error.py ========================"
python 3_error.py
echo "======================= 4_tracking / 3_plot.py ========================="
python 3_plot.py
cd ..
cd 3_detection
echo "======================= 3_detection / 4_plot.py ========================"
python 4_plot.py
cd ..