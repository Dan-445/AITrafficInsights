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


df = pd.read_excel("latest_video_detection.xlsx")
print(df)
df_person = df[df['Class'] == 'person']
df = df[df['speed'] > 11]
# df['time'] = pd.to_datetime(df['time'])
df['Converted_Time'] = df['time'].apply(lambda x: x.strftime("%I:%M:%S %p"))
df.sort_values(by='Converted_Time', inplace=True)
all_car_ids = df['ID'].unique()
all_list_data = []


# Function to remove rows where the Orientation values are not repeated for the next two rows
def remove_non_repeated_orientations(df):
    df.reset_index(drop=True, inplace=True)
    indices_to_drop = []
    for i in range(len(df) - 4):
        if not (df.iloc[i]["Orientation"] == df.iloc[i + 1]["Orientation"] and df.iloc[i + 1]["Orientation"] ==
                df.iloc[i + 2]["Orientation"] and df.iloc[i + 1]["Orientation"] ==
                df.iloc[i + 3]["Orientation"]):
            indices_to_drop.extend([i])
    for i in range(len(df) - 1, 1, -1):
        if not (df.iloc[i]["Orientation"] == df.iloc[i - 1]["Orientation"] and df.iloc[i - 1]["Orientation"] ==
                df.iloc[i - 2]["Orientation"] and df.iloc[i - 1]["Orientation"] ==
                df.iloc[i - 3]["Orientation"]):
            indices_to_drop.extend([i])
    df_cleaned = df.drop(indices_to_drop, axis=0)
    return df_cleaned


for car_id in all_car_ids:
    all_output = {}
    id_data = df[df['ID'] == car_id]
    id_data = remove_non_repeated_orientations(id_data)
    if not id_data.empty and len(id_data) > 3:
        all_output['Car_id'] = id_data['ID'].values[0]
        all_output['detected_Class'] = id_data['Class'].values[0]
        start_time = id_data['Converted_Time'].values[0]
        all_output['start_time'] = start_time
        start_orientation = id_data['Orientation'].tolist()[0]
        end_orientation = id_data['Orientation'].tolist()[-1]
        orientation = direction_mapping.get((start_orientation, end_orientation), [start_orientation, end_orientation])
        all_output['start_orientation'] = orientation[0]
        all_output['end_orientation'] = orientation[1]
        all_list_data.append(all_output)


# df_person['time'] = pd.to_datetime(df_person['time'])
df_person['Converted_Time'] = df_person['time'].apply(lambda x: x.strftime("%I:%M:%S %p"))
df_person.sort_values(by='Converted_Time', inplace=True)
all_person_ids = df_person['ID'].unique()

for person_id in all_person_ids:
        id_data = df_person[df_person['ID'] == person_id]
        id_data = remove_non_repeated_orientations(id_data)
        if not id_data.empty and len(id_data) > 2:
            all_output['Car_id'] = id_data['ID'].values[0]
            all_output['detected_Class'] = id_data['Class'].values[0]
            start_time = id_data['Converted_Time'].values[0]
            all_output['start_time'] = start_time
            start_orientation = id_data['Orientation'].tolist()[0]
            end_orientation = id_data['Orientation'].tolist()[-1]
            orientation = direction_mapping.get((start_orientation, end_orientation),
                                                [start_orientation, end_orientation])
            all_output['start_orientation'] = orientation[0]
            all_output['end_orientation'] = 'peds'
            all_list_data.append(all_output)

df_data = pd.DataFrame(all_list_data)

pivot_table = pd.pivot_table(df_data, values='detected_Class', index=['start_time'],
                             columns=['start_orientation', 'end_orientation'], aggfunc=lambda x: list(x), fill_value=0)
all_columns = [('Southbound', 'right'), ('Southbound', 'thru'), ('Southbound', 'left'), ('Southbound', 'peds'),
               ('Southbound', 'U-turn'), ('Southbound', 'Vehicle Class'), ('Westbound', 'right'), ('Westbound', 'thru'),
               ('Westbound', 'left'), ('Westbound', 'peds'), ('Westbound', 'U-turn'), ('Westbound', 'Vehicle Class'),
               ('Northbound', 'right'), ('Northbound', 'thru'), ('Northbound', 'left'), ('Northbound', 'peds'),
               ('Northbound', 'U-turn'), ('Northbound', 'Vehicle Class'), ('Eastbound', 'right'), ('Eastbound', 'thru'),
               ('Eastbound', 'left'), ('Eastbound', 'peds'), ('Eastbound', 'U-turn'), ('Eastbound', 'Vehicle Class')]
pivot_table = pivot_table.reindex(columns=all_columns, fill_value=0)
pivot_table.columns.names = ["", ""]


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
    df_data = pivot_table[direction]
    df_data = df_data.apply(map_class, axis=1)
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

    for c in [additional_cell, movement_cell]:
        c.alignment = Alignment(horizontal='center', vertical='center')
        c.fill = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")  # Gray color fill
        c.border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'),
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
for i, row in enumerate(pivot_values, start=1):
    for j, value in enumerate(row, start=1):
        worksheet.cell(row=i + len(content_data) + 2, column=j, value=value)

# Save the workbook
workbook.save('vehicle_counting_export.xlsx')
print("Data, script, and pivot table have been written to the same Excel sheet successfully.")
