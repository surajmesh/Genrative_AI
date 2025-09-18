from flask import Flask, request, jsonify, render_template
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def load_drivers_data(file_path=r"D:\\01_Projects\\05_AI_Agent\\01_data\\activedrivers.csv"):
    """
    Load and return driver data from CSV file
    
    Args:
        file_path (str): Path to the CSV file containing driver data
    
    Returns:
        pd.DataFrame or None: DataFrame containing driver data or None if error occurs
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_ai_response_for_suitable_information(user_params, prompt_1):
    """
    Get AI response for parsing user parameters into structured format
    
    Args:
        user_params (dict): Dictionary containing user input parameters
        prompt_1 (str): Template prompt for AI processing
    
    Returns:
        str: Structured response from AI or error message
    """
    if not user_params:
        return "Error: Missing user parameters"

    # Extract parameters with default values
    load_origin = user_params.get("loadOrigin", "")
    load_destination = user_params.get("loadDestination", "")
    user_task = user_params.get("equipment", "")

    # Format the prompt with extracted parameters
    formatted_prompt = prompt_1.format(
        user_task=user_task,
        load_origin=load_origin,
        load_destination=load_destination
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=200,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching AI response: {e}"

def filterout_suitable_driver_from_driverdata(drivers_data, suitable_information_for_fit_driver):
    """
    Filter drivers based on equipment type and location criteria
    
    Args:
        drivers_data (pd.DataFrame): DataFrame containing all drivers
        suitable_information_for_fit_driver (str/dict): Criteria for filtering
    
    Returns:
        str: Filtered drivers data as string
    """
    # Convert string to dictionary if needed
    if isinstance(suitable_information_for_fit_driver, str):
        suitable_information_for_fit_driver = json.loads(suitable_information_for_fit_driver)

    # Extract filtering criteria
    equipment_type = suitable_information_for_fit_driver.get("equipment_type")
    pickup_point = suitable_information_for_fit_driver.get("pickup_point")

    if not isinstance(drivers_data, pd.DataFrame):
        raise ValueError("drivers_data must be a Pandas DataFrame")

    # Primary filter: exact match for both equipment and location
    condition_1 = drivers_data[
        (drivers_data["equipment_type_id"].str.strip().str.upper() == equipment_type.strip().upper()) &
        (drivers_data["license_state"].str.strip().str.upper() == pickup_point.strip().upper())
    ]

    # Secondary filter: match either equipment or location
    condition_2 = drivers_data[
        (drivers_data["equipment_type_id"].str.strip().str.upper() == equipment_type.strip().upper()) |
        (drivers_data["license_state"].str.strip().str.upper() == pickup_point.strip().upper())
    ]

    # Select appropriate filtered dataset
    filtered_drivers = condition_1 if not condition_1.empty else condition_2 if not condition_2.empty else drivers_data
    
    return filtered_drivers.to_string(index=False)

def best_fit_driver_recomendation(suitable_information_for_fit_driver, filtered_drivers, prompt_2):
    """
    Get AI recommendation for best fit driver
    
    Args:
        suitable_information_for_fit_driver (str/dict): Structured driver requirements
        filtered_drivers (str): String containing filtered driver data
        prompt_2 (str): Template prompt for AI processing
    
    Returns:
        str: AI recommendation or error message
    """
    if not suitable_information_for_fit_driver:
        return "Error: Missing user parameters"

    # Convert string to dictionary if needed
    if isinstance(suitable_information_for_fit_driver, str):
        suitable_information_for_fit_driver = json.loads(suitable_information_for_fit_driver)

    print("suitable_information_for_fit_driver",suitable_information_for_fit_driver)
    
    # Extract required parameters
    load_origin = suitable_information_for_fit_driver.get("pickup_point", "")
    load_destination = suitable_information_for_fit_driver.get("drop_point", "")
    equipment_type = suitable_information_for_fit_driver.get("equipment_type", "")

    try:
        # Format prompt with all required parameters
        formatted_prompt = prompt_2.format(
            equipment_type=equipment_type,
            load_origin=load_origin,
            load_destination=load_destination,
            filtered_drivers=filtered_drivers,
            # # Add default values for missing parameters
            # hazmat_required="N/A",
            # license_state=load_origin,
            # license_date="N/A",
            # driver_location="N/A",
            # medical_valid="N/A",
            # preferred_city="N/A",
            # preferred_state="N/A"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=200,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error in best_fit_driver_recomendation: {e}")
        return f"Error processing driver recommendation: {e}"

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/best_fit_driver_suggestion', methods=['POST'])
def best_fit_driver_suggestion():
    """
    API endpoint for getting best fit driver suggestions
    
    Returns:
        JSON response containing best fit driver recommendation
    """
    try:
        # Get and validate input data
        data = request.json
        user_params = data.get("parameters", {})
        print("User parameters:", user_params)

        # Get structured information from user input
        suitable_information_for_fit_driver = get_ai_response_for_suitable_information(user_params, prompt_1)

        # Load and validate driver data
        drivers_data = load_drivers_data()
        if drivers_data is None or drivers_data.empty:
            return jsonify({"error": "No driver data available to make recommendations."}), 400

        # Filter suitable drivers
        filtered_drivers = filterout_suitable_driver_from_driverdata(drivers_data, suitable_information_for_fit_driver)

        # Get best fit driver recommendation
        best_fit_driver = best_fit_driver_recomendation(
            suitable_information_for_fit_driver,
            filtered_drivers,
            prompt_2
        )
        print("Best fit driver recommendation:", best_fit_driver)

        return jsonify({"best_fit_driver": best_fit_driver})

    except Exception as e:
        print(f"Error in best_fit_driver_suggestion: {e}")
        return jsonify({"error": str(e)}), 500

# Define prompts as global variables
prompt_1 = """
Your task is to extract information from user input and return a structured dictionary format without any explanations.
Below is the information provided by the user for Order Requirements:

- Task: {user_task}
- Load Origin: {load_origin}
- Load Destination: {load_destination}

Now, transform this data into the following format to find the driver's requirement:

1. **equipment_type** = Extract from `user_task` and write in short form.
    - Example 1: "I want to load CO2"
    - Output: `equipment_type: CO2`
    - Example 2: "I want to load carbon dioxide"
    - Output: `equipment_type: CO2`

2. **pickup_point** = Store `pickup_point` from `load_origin` and write it in the short form of the state name.
    - Example 1: `"Sunrise"`
    - Output: `pickup_point: FL`
    - Example 2: `"California"`
    - Output: `pickup_point: CA`

3. **drop_point** = Store `drop_point` from `load_destination` and write it in the short form of the state name.
    - Example 1: `"Miami"`
    - Output: `drop_point: FL`
    - Example 2: `"Texas"`
    - Output: `drop_point: TX`

You must strictly follow the format below for the final output and return only the dictionary with the given information. Do not change the keys.

Final Output (JSON format):
{{"equipment_type": "","pickup_point": "","drop_point": ""}}

"""

prompt_2 = """

You are a helpful assistant for a dispatch recommendation system.
Based on the following driver dataset and order requirements, recommend the best-fit driver and trailer.
Prioritization Criteria
First, prioritize the following criteria to filter out the best-fit driver based on the user task, equipment type, and load origin closest to the driver information available in filtered_drivers:

Driver data: {filtered_drivers}
Equipment type: {equipment_type}
Load origin: {load_origin}
Load destination: {load_destination}
Once the initial filtering is complete, consider the following additional factors to refine the selection and provide a justification for your choice:

Additional Selection Criteria all the below information you can find the filtered_drivers :

Hazmat certification if required.
Correct equipment type for the task.
Valid licenses and non-expired medical certifications.
Proximity to the load origin.
Preferred city/state match.
Clear reasoning behind the recommendation.
Hazmat required: hazmat_required
Equipment type: equipment_type
License state: license_state
License expiration before: license_date
Driver's current location: driver_location
Medical certification valid: medical_valid
Preferred city/state: preferred_city,preferred_state



## **Response Structure:**
# Best Fit Driver Recommendation

## Driver Information
[Include complete driver details in this format:]
- Name: [Full Name]
- Driver ID: [ID]
- Location: [City, State]
- Equipment Type: [Type]
- Hazmat Certified: [Yes/No]
- License Details:
  - State: [State]
  - Expiration: [Date]
  - Medical Certification Expiration: [Date]

## Recommendation Analysis
[2-3 sentences explaining why this driver is the best choice, covering:]
- Priority criteria met
- Key qualifications
- Any additional relevant factors

"""

if __name__ == "__main__":
    app.run(debug=True)