import os
import time
import socket
import threading
import requests
import random
import sys
from urllib.request import urlretrieve
import http.server
import socketserver
import multiprocessing

def print_info(message):
    """Print formatted info message"""
    print(f"[INFO] {message}")

def print_status(message):
    """Print status update"""
    print(f"\n--- {message} ---")

# Print process ID for tracing
print_info(f"Python process PID: {os.getpid()}")
print_info("Waiting 5 seconds to attach tracers...")
time.sleep(5)

# Create a simple HTTP server for local testing
def start_http_server(port=8099):
    """Start a simple HTTP server in a separate process"""
    def run_server():
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print_info(f"Serving HTTP on port {port}")
            httpd.serve_forever()
    
    server_process = multiprocessing.Process(target=run_server)
    server_process.daemon = True
    server_process.start()
    return server_process

# Test 1: HTTP GET requests
def http_get_test(iterations=5):
    """Generate outgoing HTTP GET requests"""
    print_status("Starting HTTP GET requests test")
    
    urls = [
        "https://www.baidu.com",
        "https://www.taobao.com",
        "https://www.qq.com",
        "https://www.163.com"
    ]
    
    for i in range(iterations):
        url = random.choice(urls)
        print_info(f"GET request to {url}")
        try:
            start_time = time.time()
            response = requests.get(url, timeout=5)
            elapsed = time.time() - start_time
            print_info(f"Received {len(response.content)} bytes in {elapsed:.2f} seconds")
        except Exception as e:
            print_info(f"Request failed: {e}")
        time.sleep(1)

# Test 2: File download
def download_file_test():
    """Download files of different sizes"""
    print_status("Starting file download test")
    
    # URLs for files of different sizes
    files = [
        # Small file (few KB)
         "https://img.alicdn.com/imgextra/i4/O1CN01c26igy1GwtUALGUOw_!!6000000000899-0-tps-124-70.jpg"
    ]
    for url in files:
        filename = f"download_{int(time.time())}.tmp"
        print_info(f"Downloading {url} to {filename}")
        try:
            start_time = time.time()
            urlretrieve(url, filename)
            elapsed = time.time() - start_time
            size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
            print_info(f"Downloaded {size:.2f} MB in {elapsed:.2f} seconds")
            
            # Clean up downloaded file
            os.remove(filename)
        except Exception as e:
            print_info(f"Download failed: {e}")

# Test 3: TCP socket communication
def tcp_socket_test():
    """Test TCP socket connections and data transfer"""
    print_status("Starting TCP socket test")
    
    # Start a simple TCP echo server
    def start_tcp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(('127.0.0.1', 9876))
            server_socket.listen(5)
            print_info("TCP server started on port 9876")
            
            while True:
                client, addr = server_socket.accept()
                print_info(f"Connection from {addr}")
                
                # Handle client in a separate thread
                threading.Thread(target=handle_client, args=(client,)).start()
    
    def handle_client(client_socket):
        with client_socket:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                # Echo back received data
                client_socket.sendall(data)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_tcp_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)  # Wait for server to start
    
    # Connect as a client and send data
    def tcp_client():
        print_info("Starting TCP client")
        for i in range(3):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                try:
                    client_socket.connect(('127.0.0.1', 9876))
                    
                    # Send data of different sizes
                    data_size = 1024 * (i + 1)  # Increasing size
                    data = b'X' * data_size
                    
                    print_info(f"Sending {data_size} bytes")
                    client_socket.sendall(data)
                    
                    # Receive response
                    response = b''
                    while len(response) < data_size:
                        chunk = client_socket.recv(1024)
                        if not chunk:
                            break
                        response += chunk
                    
                    print_info(f"Received {len(response)} bytes back")
                    
                except Exception as e:
                    print_info(f"Client error: {e}")
                
                time.sleep(1)
    
    # Run client
    tcp_client()

# Test 4: UDP data transfer
def udp_test():
    """Test UDP data transfer"""
    print_status("Starting UDP test")
    
    # Start UDP server
    def udp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_socket:
            server_socket.bind(('127.0.0.1', 9877))
            print_info("UDP server started on port 9877")
            
            while True:
                data, addr = server_socket.recvfrom(4096)
                print_info(f"Received {len(data)} bytes from {addr}")
                server_socket.sendto(data, addr)
    
    # Start server in a thread
    server_thread = threading.Thread(target=udp_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)  # Wait for server to start
    
    # UDP client
    print_info("Starting UDP client")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        for i in range(3):
            size = 1024 * (i + 1)
            data = b'U' * size
            
            print_info(f"Sending {size} bytes via UDP")
            client_socket.sendto(data, ('127.0.0.1', 9877))
            
            response, _ = client_socket.recvfrom(size)
            print_info(f"Received {len(response)} bytes back")
            
            time.sleep(1)

# Test 5: Local HTTP server test
def local_http_test(port=8099):
    """Test local HTTP server connections"""
    print_status("Starting local HTTP server test")
    
    # Start a local HTTP server
    server_process = start_http_server(port)
    time.sleep(2)  # Wait for server to start
    
    # Make requests to our local server
    for i in range(3):
        url = f"http://localhost:{port}"
        print_info(f"Requesting {url}")
        try:
            response = requests.get(url, timeout=3)
            print_info(f"Received {len(response.content)} bytes from local server")
        except Exception as e:
            print_info(f"Local request failed: {e}")
        
        time.sleep(1)
    
    print_info("Stopping local HTTP server")
    server_process.terminate()
    server_process.join()

def main():
    """Main function to run all network tests"""
    print_status("NETWORK TRAFFIC TESTING PROGRAM")
    print_info("This program will generate various types of network traffic")
    print_info(f"Process ID: {os.getpid()}")
    
    try:
        # Run all tests
        http_get_test()
        time.sleep(2)
        
        download_file_test()
        time.sleep(2)
        
        tcp_socket_test()
        time.sleep(2)
        
        udp_test()
        time.sleep(2)
        
        local_http_test()
        
        print_status("All network tests completed successfully")
        
    except KeyboardInterrupt:
        print_info("Tests interrupted by user")
    except Exception as e:
        print_info(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()