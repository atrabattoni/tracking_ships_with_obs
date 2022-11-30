"""
Download the seismological data stream over the period of interest.
"""

from obspy import read_inventory, UTCDateTime
from obspy.clients.fdsn import Client
import obsea

client = Client("RESIF")
inventory = read_inventory("../data/RR03.xml")

# Method
track = obsea.read_complex("../data/track.nc")
station = inventory[0][0]
st = obsea.load_stream(track, client, inventory, station, "*")
st.write("../data/waveform.mseed")

# Week
st = client.get_waveforms(
    network="YV",
    station="RR03",
    location="*",
    channel="*",
    starttime=UTCDateTime("2013-05-20T12:00:00"),
    endtime=UTCDateTime("2013-05-27T12:00:00"),
)
st.write("../data/waveforms.mseed")
