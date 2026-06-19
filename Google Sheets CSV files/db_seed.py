import psycopg2

# Update with your actual database credentials
db_params = {
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_PHWchgSN2ik8",
    "host": "ep-empty-butterfly-aq64mw79-pooler.c-8.us-east-1.aws.neon.tech",
    "port": "5432"
}

csv_file_path = "D&DFeats.csv"

try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # The COPY command maps the CSV columns directly to the SQL columns
    # We specify the exact column names to skip the auto-generating 'id' column
    copy_sql = """
        COPY "Feats"("Name", "Description", "Prerequisites")
        FROM STDIN WITH CSV HEADER
        DELIMITER as ','
    """

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        # copy_expert streams the file directly into PostgreSQL
        cur.copy_expert(sql=copy_sql, file=f)
        
    conn.commit()
    print("Successfully imported D&D Feats into PostgreSQL!")

    cur.close()
    conn.close()

except Exception as e:
    print(f"Error importing data: {e}")