from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("RESIF")
starttime = UTCDateTime("2013-05-20")
endtime = UTCDateTime("2013-05-29")
st = client.get_waveforms(
    "YV", "RR03", "*", "*",
    starttime, endtime, attach_response=True)

st = st.trim(
    UTCDateTime("2013-05-20T12:00:00"),
    UTCDateTime("2013-05-27T12:00:00"),
    nearest_sample=False
)

st.write("../data/waveforms.mseed")

inventory = client.get_stations(
    network="YV", station="RR03",
    starttime=starttime, endtime=endtime,
    level="response")
inventory.write("../data/inventory.xml", format="stationxml")
