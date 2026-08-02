from groq import Groq
import os
import math
from prompt import SYS_PROMPT
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None


_airports_db = None

def get_airports_db():
    global _airports_db
    if _airports_db is None:
        try:
            import airportsdata
            _airports_db = airportsdata.load('ICAO')
        except Exception as e:
            print(f"Error loading airportsdata: {e}")
            _airports_db = {}
    return _airports_db


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065  # Earth radius in NM
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    diffLong = math.radians(lon2 - lon1)

    x = math.sin(diffLong) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - (math.sin(phi1) * math.cos(phi2) * math.cos(diffLong))

    initial_bearing = math.degrees(math.atan2(x, y))
    return (initial_bearing + 360) % 360


def get_bearing_direction(bearing):
    directions = ["North", "North-Northeast", "Northeast", "East-Northeast", "East",
                  "East-Southeast", "Southeast", "South-Southeast", "South",
                  "South-Southwest", "Southwest", "West-Southwest", "West",
                  "West-Northwest", "Northwest", "North-Northwest"]
    index = int((bearing + 11.25) / 22.5) % 16
    return directions[index]


def find_nearest_airport(lat, lon):
    airports = get_airports_db()
    if not airports:
        return None

    nearest_ap = None
    min_dist = float('inf')

    for icao, ap in airports.items():
        ap_lat = ap.get('lat')
        ap_lon = ap.get('lon')
        if ap_lat is None or ap_lon is None:
            continue
        dist = haversine_distance(ap_lat, ap_lon, lat, lon)
        if dist < min_dist:
            min_dist = dist
            nearest_ap = ap

    if nearest_ap:
        bearing = calculate_bearing(nearest_ap['lat'], nearest_ap['lon'], lat, lon)
        direction = get_bearing_direction(bearing)
        return {
            "icao": nearest_ap.get('icao'),
            "name": nearest_ap.get('name'),
            "city": nearest_ap.get('city'),
            "country": nearest_ap.get('country'),
            "distance_nm": min_dist,
            "bearing_direction": direction,
            "bearing_degrees": bearing
        }
    return None


def get_response(text: str, telemetry: dict = None) -> str:
    """Send text to Groq and return text response."""
    lat = lon = alt = hdg = speed = pitch = roll = None

    if telemetry:
        for k, v in telemetry.items():
            k_lower = k.lower()
            if k_lower in ("lat", "latitude"):
                lat = v
            elif k_lower in ("lon", "longitude"):
                lon = v
            elif k_lower in ("alt", "altitude"):
                alt = v
            elif k_lower in ("hdg", "heading"):
                hdg = v
            elif k_lower in ("spd", "speed", "airspeed"):
                speed = v
            elif k_lower in ("pitch",):
                pitch = v
            elif k_lower in ("roll",):
                roll = v

    nearest_airport_info = None
    if lat is not None and lon is not None:
        nearest_airport_info = find_nearest_airport(lat, lon)

    system_content = SYS_PROMPT

    if nearest_airport_info:
        ap_name = nearest_airport_info["name"]
        ap_icao = nearest_airport_info["icao"]
        ap_city = nearest_airport_info["city"]
        dist_nm = nearest_airport_info["distance_nm"]
        direction = nearest_airport_info["bearing_direction"]

        system_content += (
            f"\nThe aircraft is currently {dist_nm:.1f} NM {direction} of "
            f"{ap_name} ({ap_icao}), located in {ap_city}."
        )
    else:
        system_content += "\nNo position data available."

    telemetry_info = []
    if alt is not None:
        alt_feet = alt * 3.28084
        telemetry_info.append(f"Altitude: {alt:.1f} meters ({alt_feet:.0f} feet)")
    if speed is not None:
        telemetry_info.append(f"Speed: {speed:.0f} knots")
    if hdg is not None:
        telemetry_info.append(f"Heading: {hdg:.1f} degrees")
    if pitch is not None and roll is not None:
        telemetry_info.append(f"Attitude: Pitch {pitch:.1f}°, Roll {roll:.1f}°")

    if telemetry_info:
        system_content += f"\nCurrent telemetry: {', '.join(telemetry_info)}."

    print(f"\n[Dynamic System Prompt Generated]:\n{system_content}\n")

    client = Groq()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]
    )
    return completion.choices[0].message.content
