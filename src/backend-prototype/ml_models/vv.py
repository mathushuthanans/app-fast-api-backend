import requests

def get_traffic_info(lat, lng):
    # Use a nearby location as destination (dummy route)
    dest_lat, dest_lng = lat + 0.01, lng + 0.01
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        "origin": f"{lat},{lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "departure_time": "now",
        "key": "AIzaSyBrcifdpry6U6Ddzuj41letrgvQgwfzBJE"  # Replace with your real API key
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()
        print("DEBUG:", data)  # Optional

        leg = data["routes"][0]["legs"][0]
        traffic_time = leg.get("duration_in_traffic", {}).get("value")  # in seconds
        normal_time = leg.get("duration", {}).get("value")  # in seconds

        return {
            "duration_normal": normal_time,
            "duration_in_traffic": traffic_time,
            "traffic_delay": traffic_time - normal_time if traffic_time and normal_time else None
        }
    except Exception as e:
        print("Error parsing traffic data:", e)
        print("Response content:", response.text)
        return None

# Example call
print(get_traffic_info(11.6643, 78.1460))
