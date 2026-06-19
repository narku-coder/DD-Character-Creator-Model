import re
import json

def convert_dataset_to_json(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = f.read()

    # Regex to capture the specific fields from the text-based dataset
    pattern = r'<USER> (.*?) <ASSISTANT>\nName: (.*?)\nSpecies: (.*?)\nClass: (.*?)\nSubclass: (.*?)\nBackground: (.*?)\nLevel: (\d+)\n\nAttributes:\nSTR: (\d+).*?\nDEX: (\d+).*?\nCON: (\d+).*?\nINT: (\d+).*?\nWIS: (\d+).*?\nCHA: (\d+).*?\n\nStarting Equipment: (.*?)\n<END>'
    
    matches = re.finditer(pattern, data, re.DOTALL)
    
    with open(output_filepath, 'w', encoding='utf-8') as out_f:
        for match in matches:
            user_prompt = match.group(1).strip()
            
            # Construct the structured JSON payload
            char_json = {
                "name": match.group(2).strip(),
                "species": match.group(3).strip(),
                "class": match.group(4).strip(),
                "subclass": match.group(5).strip(),
                "background": match.group(6).strip(),
                "level": int(match.group(7).strip()),
                "stats": {
                    "STR": int(match.group(8).strip()),
                    "DEX": int(match.group(9).strip()),
                    "CON": int(match.group(10).strip()),
                    "INT": int(match.group(11).strip()),
                    "WIS": int(match.group(12).strip()),
                    "CHA": int(match.group(13).strip())
                },
                # Split equipment by comma and remove trailing periods
                "starting_equipment": [item.strip() for item in match.group(14).strip('.').split(',')]
            }
            
            # Write the new format tailored for the JSON training pipeline
            out_f.write(f"<USER> {user_prompt} Output strictly in JSON format. <ASSISTANT>\n")
            out_f.write(json.dumps(char_json, indent=4) + "\n")
            out_f.write("<END>\n\n")

    print(f"Conversion complete! Saved to {output_filepath}")

# Execute the conversion
convert_dataset_to_json('dnd_ultimate_dataset.txt', 'dnd_json_dataset.txt')