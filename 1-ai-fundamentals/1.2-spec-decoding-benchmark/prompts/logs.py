"""
Repetitive / log-line prompts.

These have highly predictable patterns — n-gram strategies should
excel here since the text is templated and repetitive.
"""

LOG_PROMPTS = [
    (
        "logs-webserver",
        """Continue generating realistic web server access log lines in the same format:

192.168.1.100 - - [08/Jul/2026:10:15:23 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
192.168.1.101 - - [08/Jul/2026:10:15:24 +0000] "POST /api/orders HTTP/1.1" 201 89 "-" "curl/8.0"
192.168.1.102 - - [08/Jul/2026:10:15:25 +0000] "GET /api/products HTTP/1.1" 200 5678 "-" "Python/3.12"
192.168.1.100 - - [08/Jul/2026:10:15:26 +0000] "GET /api/users/42 HTTP/1.1" 200 234 "-" "Mozilla/5.0"
10.0.0.1 - - [08/Jul/2026:10:15:27 +0000] "GET /health HTTP/1.1" 200 2 "-" "kube-probe/1.28"
192.168.1.103 - - [08/Jul/2026:10:15:28 +0000] "PUT /api/users/42 HTTP/1.1" 200 56 "-" "Mozilla/5.0"
192.168.1.101 - - [08/Jul/2026:10:15:29 +0000] "GET /api/products/7 HTTP/1.1" 404 23 "-" "curl/8.0"
192.168.1.100 - - [08/Jul/2026:10:15:30 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
10.0.0.2 - - [08/Jul/2026:10:15:31 +0000] "GET /metrics HTTP/1.1" 200 45231 "-" "Prometheus/2.50"
192.168.1.104 - - [08/Jul/2026:10:15:32 +0000] "DELETE /api/orders/99 HTTP/1.1" 204 0 "-" "curl/8.0"
"""
    ),
    (
        "logs-json-entries",
        """Continue generating JSON log entries in this format for a microservices system:

{"timestamp": "2026-07-08T10:15:23.123Z", "level": "INFO", "service": "user-service", "message": "User login successful", "user_id": "u-42", "latency_ms": 145}
{"timestamp": "2026-07-08T10:15:23.456Z", "level": "WARN", "service": "order-service", "message": "Database query slow", "query": "SELECT * FROM orders WHERE user_id = ?", "duration_ms": 1250}
{"timestamp": "2026-07-08T10:15:23.789Z", "level": "ERROR", "service": "payment-service", "message": "Payment gateway timeout", "gateway": "stripe", "attempt": 2, "timeout_ms": 30000}
{"timestamp": "2026-07-08T10:15:24.001Z", "level": "INFO", "service": "user-service", "message": "Profile updated", "user_id": "u-17", "fields": ["avatar", "bio"]}
{"timestamp": "2026-07-08T10:15:24.234Z", "level": "DEBUG", "service": "auth-service", "message": "Token validation", "token_prefix": "eyJhbGci", "valid": true}
{"timestamp": "2026-07-08T10:15:24.567Z", "level": "INFO", "service": "notification-service", "message": "Email sent", "template": "welcome", "recipient": "newuser@example.com"}
{"timestamp": "2026-07-08T10:15:24.890Z", "level": "ERROR", "service": "order-service", "message": "Order creation failed", "reason": "insufficient_inventory", "product_id": "p-7", "requested": 5, "available": 2}
{"timestamp": "2026-07-08T10:15:25.111Z", "level": "INFO", "service": "user-service", "message": "User registered", "user_id": "u-103", "method": "google_oauth"}
{"timestamp": "2026-07-08T10:15:25.333Z", "level": "WARN", "service": "cache-service", "message": "Redis connection retry", "attempt": 1, "max_attempts": 3}
{"timestamp": "2026-07-08T10:15:25.555Z", "level": "INFO", "service": "api-gateway", "message": "Request routed", "path": "/api/v2/products", "upstream": "product-service-v2", "latency_ms": 89}
"""
    ),
    (
        "logs-csv-data",
        """Continue generating CSV data for server monitoring metrics:

timestamp,cpu_percent,memory_percent,disk_io_read_mb,disk_io_write_mb,net_rx_mbps,net_tx_mbps,load_1m,load_5m,load_15m
2026-07-08T10:15:00,45.2,62.1,12.3,5.1,102.4,45.2,1.2,0.9,0.7
2026-07-08T10:15:10,47.8,62.3,15.7,6.2,110.8,48.9,1.3,1.0,0.8
2026-07-08T10:15:20,52.1,62.5,18.2,7.8,125.3,52.1,1.5,1.1,0.8
2026-07-08T10:15:30,48.5,62.4,14.1,5.9,108.7,47.3,1.4,1.0,0.8
2026-07-08T10:15:40,44.9,62.2,11.8,4.8,99.2,44.1,1.2,1.0,0.8
2026-07-08T10:15:50,46.3,62.3,13.5,5.5,105.6,46.7,1.3,1.0,0.8
2026-07-08T10:16:00,51.7,62.6,17.9,7.2,122.1,51.0,1.5,1.1,0.8
2026-07-08T10:16:10,53.4,62.8,19.5,8.1,128.9,53.8,1.6,1.1,0.9
2026-07-08T10:16:20,49.8,62.5,15.3,6.1,112.4,49.2,1.4,1.1,0.9
2026-07-08T10:16:30,46.7,62.3,12.9,5.2,103.1,45.8,1.3,1.0,0.8
"""
    ),
    (
        "logs-syslog",
        """Continue generating syslog-style messages for a server monitoring system:

Jul  8 10:15:23 webserver01 nginx[1234]: 192.168.1.100 "GET /api/v2/users HTTP/2.0" 200 1234 "-" "Mozilla/5.0" 0.045
Jul  8 10:15:24 dbserver01 postgres[5678]: LOG:  checkpoint starting: time
Jul  8 10:15:25 webserver01 nginx[1234]: 10.0.0.1 "GET /health HTTP/2.0" 200 2 "-" "kube-probe/1.28" 0.002
Jul  8 10:15:26 appserver01 app[9012]: INFO  [OrderService] Processing order #ORD-7890 for user u-42
Jul  8 10:15:27 dbserver01 postgres[5678]: LOG:  checkpoint complete: wrote 42 buffers (0.3%); 0 WAL file(s) added
Jul  8 10:15:28 webserver01 nginx[1234]: 192.168.1.101 "POST /api/v2/orders HTTP/2.0" 201 89 "-" "curl/8.0" 0.234
Jul  8 10:15:29 appserver01 app[9012]: WARN  [PaymentService] Payment gateway response time 2.3s (threshold: 1.0s)
Jul  8 10:15:30 webserver01 nginx[1234]: 192.168.1.102 "GET /api/v2/products?category=electronics HTTP/2.0" 200 45231 "-" "Python/3.12" 0.567
Jul  8 10:15:31 dbserver01 postgres[5678]: ERROR:  deadlock detected, process 7890 waiting for ShareLock on transaction 12345
Jul  8 10:15:32 appserver01 app[9012]: INFO  [NotificationService] Sending email notification for order ORD-7890
"""
    ),
    (
        "logs-structured-metrics",
        """Continue generating structured metric samples in this Prometheus exposition format:

# HELP llamacpp_kv_cache_usage_ratio KV cache usage ratio
# TYPE llamacpp_kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.45
# HELP llamacpp_tokens_per_second Average tokens per second
# TYPE llamacpp_tokens_per_second gauge
llamacpp:tokens_per_second 42.7
# HELP llamacpp_active_slots Number of active processing slots
# TYPE llamacpp_active_slots gauge
llamacpp:active_slots 2
# HELP llamacpp_idle_slots Number of idle slots
# TYPE llamacpp_idle_slots gauge
llamacpp:idle_slots 6
# HELP llamacpp_total_slots Total number of slots
# TYPE llamacpp_total_slots gauge
llamacpp:total_slots 8
# HELP llamacpp_prompt_eval_count Prompt evaluation count
# TYPE llamacpp_prompt_eval_count counter
llamacpp:prompt_eval_count 128
# HELP llamacpp_prompt_eval_seconds Total prompt evaluation time
# TYPE llamacpp_prompt_eval_seconds counter
llamacpp:prompt_eval_seconds 5.2
# HELP llamacpp_tokens_predicted_count Predicted token count
# TYPE llamacpp_tokens_predicted_count counter
llamacpp:tokens_predicted_count 1024
"""
    ),
]
