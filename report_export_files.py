import pandas as pd
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import Alignment, PatternFill, Border, Side

direction_mapping = {
    ('North', 'North'): ['Northbound', 'thru'],
    ('south', 'south'): ['Southbound', 'thru'],
    ('East', 'East'): ['Eastbound', 'thru'],
    ('West', 'West'): ['Westbound', 'thru'],
    ('south', 'West'): ['Southbound', 'left'],
    ('North', 'south'): ['Northbound', 'U-turn'],
    ('North', 'East'): ['Northbound', 'left'],
    ('North', 'West'): ['Northbound', 'right'],
    ('south', 'North'): ['Southbound', 'U-turn'],
    ('south', 'East'): ['Southbound', 'right'],
    ('East', 'North'): ['Eastbound', 'right'],
    ('East', 'south'): ['Eastbound', 'left'],
    ('East', 'West'): ['Eastbound', 'U-turn'],
    ('West', 'North'): ['Westbound', 'left'],
    ('West', 'south'): ['Westbound', 'right'],
    ('West', 'East'): ['Westbound', 'U-turn']
}


def convert_time_format(time_str):
    time_parts = time_str.split(':')
    total_seconds = int(time_parts[0]) + float(time_parts[1])
    converted_time = (datetime.min + timedelta(seconds=total_seconds)).time()
    return converted_time.strftime("%I:%M:%S %p")


df_vehicle_detection = pd.read_excel("latest_video_detection.xlsx")
df_person = df_vehicle_detection[df_vehicle_detection['Class'] == 'person']
df_vehicle_detection = df_vehicle_detection[df_vehicle_detection['speed'] > 11]
df_vehicle_detection['Converted_Time'] = df_vehicle_detection['time'].apply(lambda x: x.strftime("%I:%M:%S %p"))
df_vehicle_detection.sort_values(by='Converted_Time', inplace=True)
all_car_ids = df_vehicle_detection['ID'].unique()
detection_tracking_list = []


# Function to remove rows where the Orientation values are not repeated for the next two rows
def remove_non_repeated_orientations(df_car_tracking):
    df_car_tracking.reset_index(drop=True, inplace=True)
    indices_to_drop = []
    for row in range(len(df_car_tracking) - 2):
        if not (df_car_tracking.iloc[row]["Orientation"] == df_car_tracking.iloc[row + 1]["Orientation"]):
            indices_to_drop.extend([row])
    for row in range(len(df_car_tracking) - 1, 1, -1):
        if not (df_car_tracking.iloc[row]["Orientation"] == df_car_tracking.iloc[row - 1]["Orientation"]):
            indices_to_drop.extend([row])
    df_cleaned = df_car_tracking.drop(indices_to_drop, axis=0)
    return df_cleaned


for car_id in all_car_ids:
    vehicle_tracking_output = {}
    vehicle_tracking_dataset = df_vehicle_detection[df_vehicle_detection['ID'] == car_id]
    # vehicle_tracking_dataset = remove_non_repeated_orientations(vehicle_tracking_dataset)
    if not vehicle_tracking_dataset.empty:
        vehicle_tracking_output['Car_id'] = vehicle_tracking_dataset['ID'].values[0]
        vehicle_tracking_output['detected_Class'] = vehicle_tracking_dataset['Class'].values[0]
        start_time = vehicle_tracking_dataset['Converted_Time'].values[0]
        vehicle_tracking_output['start_time'] = start_time
        start_orientation = vehicle_tracking_dataset['Orientation'].tolist()[0]
        end_orientation = vehicle_tracking_dataset['Orientation'].tolist()[-1]
        orientation = direction_mapping.get((start_orientation, end_orientation), [start_orientation, end_orientation])
        vehicle_tracking_output['start_orientation'] = orientation[0]
        vehicle_tracking_output['end_orientation'] = orientation[1]
        detection_tracking_list.append(vehicle_tracking_output)

# df_person['time'] = pd.to_datetime(df_person['time'])
df_person['Converted_Time'] = df_person['time'].apply(lambda time: time.strftime("%I:%M:%S %p"))
df_person.sort_values(by='Converted_Time', inplace=True)
detected_person_ids = df_person['ID'].unique()

for person_id in detected_person_ids:
    person_tracking_output = {}
    person_tracking_dataset = df_person[df_person['ID'] == person_id]
    person_tracking_dataset = remove_non_repeated_orientations(person_tracking_dataset)
    if not person_tracking_dataset.empty and len(person_tracking_dataset) > 2:
        person_tracking_output['Car_id'] = person_tracking_dataset['ID'].values[0]
        person_tracking_output['detected_Class'] = person_tracking_dataset['Class'].values[0]
        start_time = person_tracking_dataset['Converted_Time'].values[0]
        person_tracking_output['start_time'] = start_time
        start_orientation = person_tracking_dataset['Orientation'].tolist()[0]
        end_orientation = person_tracking_dataset['Orientation'].tolist()[-1]
        orientation = direction_mapping.get((start_orientation, end_orientation),
                                            [start_orientation, end_orientation])
        person_tracking_output['start_orientation'] = orientation[0]
        person_tracking_output['end_orientation'] = 'peds'
        detection_tracking_list.append(person_tracking_output)

direction_detected_dataset = pd.DataFrame(detection_tracking_list)

pivot_table = pd.pivot_table(direction_detected_dataset, values='detected_Class', index=['start_time'],
                             columns=['start_orientation', 'end_orientation'], aggfunc=lambda x: list(x), fill_value=0)
all_columns = [('Southbound', 'right'), ('Southbound', 'thru'), ('Southbound', 'left'), ('Southbound', 'peds'),
               ('Southbound', 'U-turn'), ('Southbound', 'Vehicle Class'), ('Westbound', 'right'), ('Westbound', 'thru'),
               ('Westbound', 'left'), ('Westbound', 'peds'), ('Westbound', 'U-turn'), ('Westbound', 'Vehicle Class'),
               ('Northbound', 'right'), ('Northbound', 'thru'), ('Northbound', 'left'), ('Northbound', 'peds'),
               ('Northbound', 'U-turn'), ('Northbound', 'Vehicle Class'), ('Eastbound', 'right'), ('Eastbound', 'thru'),
               ('Eastbound', 'left'), ('Eastbound', 'peds'), ('Eastbound', 'U-turn'), ('Eastbound', 'Vehicle Class')]
pivot_table = pivot_table.reindex(columns=all_columns, fill_value=0)
pivot_table.columns.names = ["", ""]
column_dict = {col: 0 for col in all_columns}
new_row = pd.Series(column_dict, name='total_count')
pivot_table = pd.concat([pivot_table, new_row.to_frame().T])


def map_class(row):
    all_classes = []
    for direction in ['right', 'thru', 'left', 'U-turn']:
        if isinstance(row[direction], list):
            all_classes.extend(row[direction])
            row[direction] = len(row[direction])
    row['Vehicle Class'] = ','.join(all_classes)
    if isinstance(row["peds"], list):
        row["peds"] = len(row["peds"])
    return row


for direction in ['Southbound', 'Westbound', 'Northbound', 'Eastbound']:
    direction_columns ={}
    df_data = pivot_table[direction]
    df_data = df_data.apply(map_class, axis=1)
    for vehicle_direction in ['right', 'thru', 'left','peds', 'U-turn']:
          direction_columns[vehicle_direction] = df_data[vehicle_direction].sum()
    df_data.loc['total_count'] = direction_columns
    pivot_table[direction] = df_data

# Create a new workbook
workbook = Workbook()
worksheet = workbook.active

# Data and formatting for the first set of cells
header_data = ['Start Date:', 'Start Time:', 'Site Code:', 'Comment 1:', 'Comment 2:', 'Comment 3:', 'Comment 4:']
for row, data in enumerate(header_data, start=1):
    cell = worksheet.cell(row, 1, data)
    cell.alignment = Alignment(horizontal='right', vertical='center')
    worksheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)

# Data and formatting for the second set of cells
content_data = ['', '6:00:00 AM', '', 'Default Comments', 'Change These in The Preferences Window',
                'Select File/Preference in the Main Screen', 'Then Click the Comments Tab']
for row, data in enumerate(content_data, start=1):
    cell = worksheet.cell(row, 3, data)
    cell.alignment = Alignment(horizontal='left', vertical='center')
    worksheet.merge_cells(start_row=row, start_column=3, end_row=row, end_column=22)

# Data and formatting for the columns in rows 8 and 9
header_data = ['Southbound', 'Westbound', 'Northbound', 'Eastbound']
for col, data in enumerate(header_data, start=1):
    col += (col - 1) * 5
    cell = worksheet.cell(8, col + 1, data)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")  # Gray color fill
    cell.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'),
                         bottom=Side(style='thin'))
    worksheet.merge_cells(start_row=8, start_column=col + 1, end_row=8, end_column=col + 4)
    additional_cell = worksheet.cell(8, col + 5, 'Additional')
    movement_cell = worksheet.cell(8, col + 6, 'Movement')

    for cell in [additional_cell, movement_cell]:
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")  # Gray color fill
        cell.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'),
                             bottom=Side(style='thin'))

header_data = ['Right', 'Thru', 'Left', 'Peds', 'U-Turns', 'Vehicle Class'] * 4
cell = worksheet.cell(9, 1, 'Start Time')
cell.alignment = Alignment(horizontal='center', vertical='center')
cell.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")  # Gray color fill
cell.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'),
                     bottom=Side(style='thin'))

for col, data in enumerate(header_data, start=1):
    cell = worksheet.cell(9, col + 1, data)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    cell.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")  # Gray color fill
    cell.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'),
                         bottom=Side(style='thin'))

# Adjust row heights
worksheet.row_dimensions[8].height = 30
worksheet.row_dimensions[9].height = 30

# Convert the pivot table to a list of lists
pivot_values = pivot_table.reset_index().values.tolist()
# Write the script and the pivot table to the same sheet
for row_data, row in enumerate(pivot_values, start=1):
    for column_data, value in enumerate(row, start=1):
        worksheet.cell(row=row_data + len(content_data) + 2, column=column_data, value=value)

# Save the workbook
workbook.save('vehicle_counting_export.xlsx')
print("Pivot table have been written to the same Excel sheet successfully")
