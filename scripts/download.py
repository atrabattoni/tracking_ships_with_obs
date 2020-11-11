from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("RESIF")
starttime = UTCDateTime("2013-05-20")
endtime = UTCDateTime("2013-05-29")
st = client.get_waveforms(
    "YV", "RR03", "*", "*",
    starttime, endtime, attach_response=True)
st.write("waveforms.mseed")

inventory = client.get_stations(
    network="YV", station="RR03",
    starttime=starttime, endtime=endtime,
    level="response")
inventory.write("inventory.xml", format="stationxml")
