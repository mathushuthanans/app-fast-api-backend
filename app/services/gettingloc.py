import geocoder
from geopy.geocoders import Nominatim


def addr():

    # Create geopy geolocator with proper User-Agent (very important!)
    geolocator = Nominatim(user_agent="backend-prototype-app (your_email@example.com)")

    # Get IP-based location (may not be very accurate)
    g = geocoder.ip('me')

    if g.ok and g.latlng:
        lat = g.lat
        lng = g.lng

        print("Latitude:", lat)
        print("Longitude:", lng)

        # Now use geopy for reverse geocoding (NOT geocoder.osm)
        location = geolocator.reverse((lat, lng), language='en')

        if location:
            return location.address
        else:
            return "Address not found."
    else:
        return "Could not determine location."
