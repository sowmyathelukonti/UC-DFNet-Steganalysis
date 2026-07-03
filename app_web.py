import os
import math
import time
import random
import threading
import collections
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# System State Variables
processes = [
    {"pid": 104, "name": "system_kernel.exe", "cpu": 1.2, "memory": "45 MB", "io_rate": "0.1 KB/s", "entropy": 3.42, "status": "Running", "risk": "Safe"},
    {"pid": 1120, "name": "explorer.exe", "cpu": 0.8, "memory": "112 MB", "io_rate": "0.5 KB/s", "entropy": 4.15, "status": "Running", "risk": "Safe"},
    {"pid": 2408, "name": "chrome.exe", "cpu": 3.5, "memory": "320 MB", "io_rate": "12.4 KB/s", "entropy": 4.62, "status": "Running", "risk": "Safe"},
    {"pid": 3120, "name": "teams.exe", "cpu": 1.1, "memory": "145 MB", "io_rate": "1.2 KB/s", "entropy": 4.51, "status": "Running", "risk": "Safe"},
    {"pid": 4012, "name": "word.exe", "cpu": 0.4, "memory": "85 MB", "io_rate": "2.1 KB/s", "entropy": 4.38, "status": "Running", "risk": "Safe"}
]

honeytokens = [
    {"id": 1, "filename": "passwords.txt", "path": "C:\\Users\\hp 4021\\Documents\\passwords.txt", "type": "Text Decoy", "size": "1.2 KB", "status": "Untouched", "last_accessed": "Never"},
    {"id": 2, "filename": "financial_records.xlsx", "path": "C:\\Users\\hp 4021\\Desktop\\financial_records.xlsx", "type": "Spreadsheet Decoy", "size": "45.8 KB", "status": "Untouched", "last_accessed": "Never"},
    {"id": 3, "filename": "db_backup.sql", "path": "C:\\Users\\hp 4021\\AppData\\db_backup.sql", "type": "Database Decoy", "size": "1.2 MB", "status": "Untouched", "last_accessed": "Never"}
]

security_logs = [
    {"timestamp": time.strftime("%H:%M:%S"), "source": "SYSTEM", "event": "EDR Engine Initialized successfully.", "level": "INFO"},
    {"timestamp": time.strftime("%H:%M:%S"), "source": "DECEPTION_ENGINE", "event": "3 Honeytoken decoys deployed to vulnerable folders.", "level": "INFO"},
    {"timestamp": time.strftime("%H:%M:%S"), "source": "ENTROPY_MONITOR", "event": "Real-time Shannon entropy scanning activated. Threshold set to 6.5.", "level": "INFO"}
]

# Risk configuration
entropy_threshold = 6.5
simulation_active = False
simulated_pid = None

def calculate_shannon_entropy(data: str) -> float:
    """Computes Shannon entropy of a string."""
    if not data:
        return 0.0
    counts = collections.Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 2)

def log_event(source, event, level="INFO"):
    security_logs.insert(0, {
        "timestamp": time.strftime("%H:%M:%S"),
        "source": source,
        "event": event,
        "level": level
    })
    # Keep only recent 50 logs
    if len(security_logs) > 50:
        security_logs.pop()

def run_ransomware_simulation():
    global simulation_active, simulated_pid
    sim_pid = random.randint(5000, 9999)
    simulated_pid = sim_pid
    
    # Add simulation process to process explorer
    sim_process = {
        "pid": sim_pid,
        "name": "wanacry_cryptor.exe",
        "cpu": 15.6,
        "memory": "12 MB",
        "io_rate": "0 KB/s",
        "entropy": 4.10,
        "status": "Running",
        "risk": "Medium"
    }
    processes.append(sim_process)
    
    log_event("ENTROPY_MONITOR", f"New suspicious process detected: {sim_process['name']} (PID: {sim_pid})", "WARN")
    
    steps = [
        ("Scanning C:\\Users\\hp 4021\\Documents...", "normal_text_sample_12345", 3.82, "Safe", 5),
        ("Opening passwords.txt decoy...", "normal_text_sample_54321", 4.10, "Safe", 8),
        ("Beginning fast encryption of documents...", "F83jD#ks*dfjL9@sDfj%ks", 5.92, "Medium", 25),
        ("Encrypting db_backup.sql decoy...", "8c0a9dfb2e3f4e5a6b7c8d9e0f", 7.84, "Critical", 65)
    ]
    
    for log_msg, sample_data, mock_entropy, risk_level, io_speed in steps:
        if not simulation_active:
            break
            
        # Update process stats
        sim_process["entropy"] = mock_entropy
        sim_process["io_rate"] = f"{io_speed} KB/s"
        sim_process["cpu"] = round(random.uniform(25.0, 45.0), 1)
        sim_process["risk"] = risk_level
        
        # Calculate real entropy of sample data
        real_entropy = calculate_shannon_entropy(sample_data)
        
        log_event("ENTROPY_MONITOR", f"{log_msg} [Write Entropy: {mock_entropy} (Calculated: {real_entropy})]", "WARN" if mock_entropy < 6.5 else "ALERT")
        
        # Check Layer 2 trigger: if accessing decoy
        if "passwords.txt" in log_msg or "db_backup.sql" in log_msg:
            # Trigger honeytoken access
            decoy_name = "passwords.txt" if "passwords.txt" in log_msg else "db_backup.sql"
            for token in honeytokens:
                if token["filename"] == decoy_name:
                    token["status"] = "Compromised"
                    token["last_accessed"] = time.strftime("%H:%M:%S")
            log_event("DECEPTION_ENGINE", f"Honeytoken '{decoy_name}' access detected by PID {sim_pid}! Attacker Source IP: 192.168.1.{random.randint(100, 254)}", "ALERT")
            
        # Check Layer 1 trigger: Entropy threshold crossed
        if mock_entropy >= entropy_threshold:
            log_event("RISK_ENGINE", f"CRITICAL THREAT: Write Entropy {mock_entropy} exceeded threshold {entropy_threshold}!", "ALERT")
            # Auto terminate
            sim_process["status"] = "Terminated"
            sim_process["risk"] = "Terminated"
            sim_process["cpu"] = 0.0
            sim_process["io_rate"] = "0 KB/s"
            log_event("SYSTEM", f"PROCESS TERMINATED BY EDR ENGINE: PID {sim_pid} ({sim_process['name']})", "INFO")
            simulation_active = False
            break
            
        time.sleep(2.5)
        
    simulation_active = False

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global entropy_threshold
    active_processes = len([p for p in processes if p["status"] == "Running"])
    compromised_decoys = len([t for t in honeytokens if t["status"] == "Compromised"])
    
    threat_score = 0
    if compromised_decoys > 0:
        threat_score += 45
    if any(p["risk"] == "Critical" and p["status"] == "Running" for p in processes):
        threat_score += 50
    elif any(p["risk"] == "Medium" and p["status"] == "Running" for p in processes):
        threat_score += 25
        
    threat_score = min(100, threat_score)
    
    return jsonify({
        "status": "SECURE" if threat_score < 40 else ("WARNING" if threat_score < 75 else "BREACHED"),
        "threat_score": threat_score,
        "active_processes": active_processes,
        "compromised_decoys": compromised_decoys,
        "entropy_threshold": entropy_threshold
    })

@app.route('/api/processes', methods=['GET'])
def get_processes():
    return jsonify(processes)

@app.route('/api/honeytokens', methods=['GET'])
def get_honeytokens():
    return jsonify(honeytokens)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify(security_logs)

@app.route('/api/deploy_honeytoken', methods=['POST'])
def deploy_honeytoken():
    filename = request.form.get("filename")
    path = request.form.get("path")
    decoy_type = request.form.get("type", "Text Decoy")
    size = request.form.get("size", "2.4 KB")
    
    if not filename or not path:
        return jsonify({"success": False, "error": "Missing decoy file configurations."}), 400
        
    new_token = {
        "id": len(honeytokens) + 1,
        "filename": filename,
        "path": path,
        "type": decoy_type,
        "size": size,
        "status": "Untouched",
        "last_accessed": "Never"
    }
    honeytokens.append(new_token)
    log_event("DECEPTION_ENGINE", f"New Honeytoken decoy deployed: {filename} in {path}", "INFO")
    return jsonify({"success": True})

@app.route('/api/simulate_ransomware', methods=['POST'])
def trigger_simulation():
    global simulation_active
    if simulation_active:
        return jsonify({"success": False, "error": "Simulation already in progress."}), 400
        
    simulation_active = True
    # Reset honeytokens accessed status for demo
    for token in honeytokens:
        token["status"] = "Untouched"
        token["last_accessed"] = "Never"
        
    # Start thread
    thread = threading.Thread(target=run_ransomware_simulation)
    thread.daemon = True
    thread.start()
    
    log_event("SYSTEM", "User triggered zero-day ransomware attack simulation.", "INFO")
    return jsonify({"success": True})

@app.route('/api/trigger_honeytoken_alert', methods=['POST'])
def trigger_decoy():
    decoy_id = int(request.form.get("id"))
    for token in honeytokens:
        if token["id"] == decoy_id:
            token["status"] = "Compromised"
            token["last_accessed"] = time.strftime("%H:%M:%S")
            mock_ip = f"192.168.1.{random.randint(100, 254)}"
            log_event("DECEPTION_ENGINE", f"ALERT: Unauthorized access on honeytoken '{token['filename']}'! Source IP: {mock_ip}", "ALERT")
            return jsonify({"success": True})
            
    return jsonify({"success": False, "error": "Honeytoken not found."}), 400

@app.route('/api/terminate_process', methods=['POST'])
def terminate_process():
    pid = int(request.form.get("pid"))
    for p in processes:
        if p["pid"] == pid:
            p["status"] = "Terminated"
            p["risk"] = "Terminated"
            p["cpu"] = 0.0
            p["io_rate"] = "0 KB/s"
            log_event("SYSTEM", f"User manually terminated process: {p['name']} (PID: {pid})", "INFO")
            return jsonify({"success": True})
            
    return jsonify({"success": False, "error": "Process not found."}), 400

@app.route('/api/update_threshold', methods=['POST'])
def update_threshold():
    global entropy_threshold
    threshold = float(request.form.get("threshold", "6.5"))
    entropy_threshold = threshold
    log_event("ENTROPY_MONITOR", f"Shannon entropy monitoring threshold set to {threshold}.", "INFO")
    return jsonify({"success": True})

if __name__ == '__main__':
    # Clean up Wanacry simulation from processes list on restart
    processes = [p for p in processes if "wanacry" not in p["name"].lower()]
    app.run(host='0.0.0.0', port=5000, debug=False)
