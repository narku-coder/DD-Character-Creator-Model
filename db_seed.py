import psycopg2

# Update with your actual database credentials
db_params = {
    "dbname": "your_db_name",
    "user": "your_db_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

csv_file_path = "class_features.csv"

try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # The COPY command maps the CSV columns directly to the SQL columns
    # We specify the exact column names to skip the auto-generating 'id' column
    copy_sql = """
        COPY "ClassFeatures"("Name", "Description", "Class", "Subclass", "Level")
        FROM STDIN WITH CSV HEADER
        DELIMITER as ','
    """

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        # copy_expert streams the file directly into PostgreSQL
        cur.copy_expert(sql=copy_sql, file=f)
        
    conn.commit()
    print("Successfully imported D&D Class Features into PostgreSQL!")

    cur.close()
    conn.close()

except Exception as e:
    print(f"Error importing data: {e}")