from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("RESIF")

# Metadata
inventory = client.get_stations(
    network="YV",
    station="RR03",
    starttime=UTCDateTime("2012-10-01"),
    endtime=UTCDateTime("2013-11-30"),
    level="response",
)
inventory.write("../data/RR03.xml", format="stationxml")
