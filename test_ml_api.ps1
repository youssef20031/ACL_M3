# PowerShell Script to Test ML API Endpoints
# Usage: .\test_ml_api.ps1

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "FPL ML API TEST SUITE" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$baseUrl = "http://localhost:8000"

# Test 1: ML Status
Write-Host "Test 1: ML Status Endpoint" -ForegroundColor Yellow
Write-Host "GET $baseUrl/api/ml/status`n" -ForegroundColor Gray

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/ml/status" -Method Get
    Write-Host "✅ ML Status:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host "   Make sure API server is running: python api_main.py" -ForegroundColor Yellow
}

Write-Host "`n------------------------------------------------------------`n"

# Test 2: Predict Single Player
Write-Host "Test 2: Predict Single Player (Mohamed Salah)" -ForegroundColor Yellow
Write-Host "POST $baseUrl/api/ml/predict/player`n" -ForegroundColor Gray

$body = @{
    player_name = "Mohamed Salah"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/ml/predict/player" `
                                  -Method Post `
                                  -ContentType "application/json" `
                                  -Body $body
    
    Write-Host "✅ Prediction for Mohamed Salah:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}

Write-Host "`n------------------------------------------------------------`n"

# Test 3: Top Performers
Write-Host "Test 3: Top 5 Forwards" -ForegroundColor Yellow
Write-Host "POST $baseUrl/api/ml/predict/top-performers`n" -ForegroundColor Gray

$body = @{
    position = "FWD"
    top_k = 5
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/ml/predict/top-performers" `
                                  -Method Post `
                                  -ContentType "application/json" `
                                  -Body $body
    
    Write-Host "✅ Top 5 Forwards:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}

Write-Host "`n------------------------------------------------------------`n"

# Test 4: Best Value
Write-Host "Test 4: Best Value Midfielders (£8.0m max)" -ForegroundColor Yellow
Write-Host "POST $baseUrl/api/ml/predict/best-value`n" -ForegroundColor Gray

$body = @{
    position = "MID"
    max_price = 8.0
    top_k = 5
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/ml/predict/best-value" `
                                  -Method Post `
                                  -ContentType "application/json" `
                                  -Body $body
    
    Write-Host "✅ Best Value Midfielders:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "TEST SUITE COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan
