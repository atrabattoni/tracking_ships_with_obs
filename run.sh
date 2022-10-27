mkdir -p data
mkdir -p figs
cd 0_data
python 1_station.py
python 2_tracks.py
python 3_streams.py
cd ..
cd "1_exp&obs"
python 1_spectrogram.py
python 2_ais.py
cd ..
cd 2_method
python 1_adjust_model.py
python 1_branch_association.py
python 2_direction.py
python 2_distance.py
cd ..
cd 3_detection
python 1_direction.py
python 1_distance.py
python 2_error.py
cd ..
cd 4_tracking 
python 1_segments.py
python 2_process.py
python 3_error.py
python 3_plot.py
cd ..
cd 3_detection
python 4_plot.py
cd ..