import os
import pymysql

def setup_mysql():
    root_password = os.environ.get("MYSQL_ROOT_PASSWORD", "Admin@123")
    app_user = os.environ.get("MYSQL_USER", "anprx_app")
    app_password = os.environ.get("MYSQL_PASSWORD", "AnprxAppSecurePass2026!")
    db_name = os.environ.get("MYSQL_DATABASE", "anprx")
    host = os.environ.get("MYSQL_HOST", "localhost")
    port = int(os.environ.get("MYSQL_PORT", 3306))

    print(f"Connecting to MySQL on {host}:{port} as root...")
    conn = pymysql.connect(
        host=host,
        port=port,
        user="root",
        password=root_password
    )
    cur = conn.cursor()

    print(f"Creating database `{db_name}` if not exists...")
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")

    print(f"Ensuring user `{app_user}`@`%` and `{app_user}`@`localhost`...")
    for host_spec in ["localhost", "%"]:
        cur.execute(f"CREATE USER IF NOT EXISTS '{app_user}'@'{host_spec}' IDENTIFIED BY '{app_password}';")
        cur.execute(f"ALTER USER '{app_user}'@'{host_spec}' IDENTIFIED BY '{app_password}';")
        cur.execute(f"GRANT ALL PRIVILEGES ON `{db_name}`.* TO '{app_user}'@'{host_spec}';")

    cur.execute("FLUSH PRIVILEGES;")
    conn.close()

    print(f"Testing connection as `{app_user}` to database `{db_name}`...")
    app_conn = pymysql.connect(
        host=host,
        port=port,
        user=app_user,
        password=app_password,
        database=db_name
    )
    print(f"Successfully connected as {app_user} to database {db_name}!")
    app_conn.close()

if __name__ == "__main__":
    setup_mysql()
