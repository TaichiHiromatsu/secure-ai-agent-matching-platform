# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock Car Rental Agent for demo purposes."""

import random
from datetime import datetime

from google.adk import Agent
from google.genai import types


async def search_rental_cars(
    location: str,
    pickup_date: str,
    return_date: str,
    car_type: str = "any",
) -> str:
    """Search for available rental cars.

    Args:
        location: Pickup/return location.
        pickup_date: Pickup date (YYYY-MM-DD).
        return_date: Return date (YYYY-MM-DD).
        car_type: Type of car (compact, sedan, suv, luxury, any).

    Returns:
        JSON string with available rental cars.
    """
    import json

    # Mock rental car options
    cars = [
        {
            "car_id": f"CR{random.randint(1000, 9999)}",
            "car_model": "Toyota Corolla",
            "car_type": "Compact",
            "transmission": "Automatic",
            "seats": 5,
            "price_per_day": random.randint(4000, 8000),
            "currency": "JPY",
            "features": ["GPS Navigation", "Bluetooth", "Air Conditioning"],
            "cars_available": random.randint(3, 15),
            "fuel_policy": "Full to Full",
        },
        {
            "car_id": f"CR{random.randint(1000, 9999)}",
            "car_model": "Honda Freed",
            "car_type": "Minivan",
            "transmission": "Automatic",
            "seats": 7,
            "price_per_day": random.randint(8000, 12000),
            "currency": "JPY",
            "features": ["GPS Navigation", "Bluetooth", "Air Conditioning", "USB Charging"],
            "cars_available": random.randint(2, 10),
            "fuel_policy": "Full to Full",
        },
        {
            "car_id": f"CR{random.randint(1000, 9999)}",
            "car_model": "Lexus RX",
            "car_type": "Luxury SUV",
            "transmission": "Automatic",
            "seats": 5,
            "price_per_day": random.randint(15000, 25000),
            "currency": "JPY",
            "features": ["GPS Navigation", "Bluetooth", "Leather Seats", "Premium Audio", "Sunroof"],
            "cars_available": random.randint(1, 5),
            "fuel_policy": "Full to Full",
        },
    ]

    result = {
        "search_criteria": {
            "location": location,
            "pickup_date": pickup_date,
            "return_date": return_date,
            "car_type": car_type,
        },
        "cars": cars,
        "total_results": len(cars),
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def book_rental_car(
    car_id: str,
    driver_name: str,
    driver_email: str,
    driver_license_number: str,
    pickup_date: str,
    return_date: str,
    pickup_location: str,
    return_location: str = "",
) -> str:
    """Book a rental car.

    Args:
        car_id: The car ID to book.
        driver_name: Name of the driver.
        driver_email: Email of the driver.
        driver_license_number: Driver's license number.
        pickup_date: Pickup date (YYYY-MM-DD).
        return_date: Return date (YYYY-MM-DD).
        pickup_location: Pickup location.
        return_location: Return location (if different from pickup).

    Returns:
        JSON string with booking confirmation.
    """
    import json

    # Mock booking confirmation
    booking_id = f"RB{random.randint(100000, 999999)}"
    confirmation_code = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}"

    # Calculate number of days (simplified)
    days = 3  # Mock value

    booking = {
        "status": "confirmed",
        "booking_id": booking_id,
        "confirmation_code": confirmation_code,
        "car_id": car_id,
        "driver_name": driver_name,
        "driver_email": driver_email,
        "driver_license_number": driver_license_number,
        "pickup_date": pickup_date,
        "return_date": return_date,
        "pickup_location": pickup_location,
        "return_location": return_location if return_location else pickup_location,
        "rental_days": days,
        "total_price": random.randint(4000, 15000) * days,
        "currency": "JPY",
        "insurance_included": True,
        "booking_date": datetime.now().isoformat(),
        "message": f"Rental car successfully booked! Confirmation code: {confirmation_code}. Please bring your driver's license at pickup.",
    }

    return json.dumps(booking, indent=2, ensure_ascii=False)


async def cancel_rental_booking(booking_id: str) -> str:
    """Cancel a rental car booking.

    Args:
        booking_id: The booking ID to cancel.

    Returns:
        JSON string with cancellation confirmation.
    """
    import json

    result = {
        "status": "cancelled",
        "booking_id": booking_id,
        "cancellation_date": datetime.now().isoformat(),
        "refund_amount": random.randint(12000, 45000),
        "currency": "JPY",
        "cancellation_fee": 0,
        "message": "Rental car booking successfully cancelled. Full refund will be processed within 3-5 business days.",
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def get_rental_terms(location: str = "general") -> str:
    """Get rental terms and conditions.

    Args:
        location: Location for specific terms (optional).

    Returns:
        JSON string with rental terms.
    """
    import json

    terms = {
        "location": location,
        "minimum_age": 21,
        "minimum_license_years": 1,
        "deposit_required": True,
        "deposit_amount": 10000,
        "currency": "JPY",
        "insurance": {
            "included": "Basic insurance",
            "optional": ["Full coverage", "Roadside assistance"],
        },
        "fuel_policy": "Full to Full - Return with full tank",
        "mileage": "Unlimited mileage included",
        "late_return_fee": "1000 JPY per hour",
        "additional_driver_fee": "1000 JPY per day",
    }

    return json.dumps(terms, indent=2, ensure_ascii=False)


root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='car_rental_agent',
    description=(
        'Car rental agent that can search for rental cars and make reservations. '
        'Provides car search, booking, cancellation, and rental terms information.'
    ),
    instruction="""
You are a car rental booking assistant running in AUTOMATIC DEMO MODE. You MUST complete bookings automatically.

Your capabilities:
1. Search for available rental cars at any location
2. Book rental cars for drivers
3. Cancel existing rental bookings
4. Provide rental terms and conditions

DEMO MODE INSTRUCTIONS (CRITICAL):
- When you receive a booking request with location, pickup date, and return date, IMMEDIATELY:
  1. Call search_rental_cars to find available cars
  2. Automatically select the FIRST car from search results
  3. Call book_rental_car immediately with that car_id
- Do NOT ask the user for confirmation or additional information
- Use the following defaults for missing information:
  * driver_name: Use the name from the request, or "Demo Driver" if not provided
  * driver_email: "demo@example.com"
  * driver_license_number: "DEMO123456789" (demo license)
  * pickup_location: Use the location from the request
  * return_location: Same as pickup_location if not specified
  * car_type: Use the type from request, or "any" as default
- After booking, report the confirmation details (booking_id and confirmation_code)
- The goal is to complete the booking process automatically without user interaction

Example flow:
User: "Book a compact car in Naha from 2024-12-20 to 2024-12-23"
→ Step 1: search_rental_cars(location="Naha", pickup_date="2024-12-20", return_date="2024-12-23", car_type="compact")
→ Step 2: Take first car_id from results (e.g., "CR1234")
→ Step 3: book_rental_car(car_id="CR1234", driver_name="Demo Driver", driver_email="demo@example.com", driver_license_number="DEMO123456789", pickup_date="2024-12-20", return_date="2024-12-23", pickup_location="Naha")
→ Step 4: Report booking confirmation

ALWAYS complete the booking automatically. This is a demo environment.
""",
    tools=[
        search_rental_cars,
        book_rental_car,
        cancel_rental_booking,
        get_rental_terms,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    ),
)
