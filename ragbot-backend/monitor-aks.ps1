# Start a new PowerShell window for pod logs
Start-Process powershell -ArgumentList "-NoExit", "-Command", "kubectl logs -n ragbot -l app=ragbot --follow"

# Start a new PowerShell window for pod resource usage
Start-Process powershell -ArgumentList "-NoExit", "-Command", "while (`$true) { Clear-Host; Write-Host 'Pod Resource Usage:'; kubectl top pods -n ragbot; Write-Host '`nNode Resource Usage:'; kubectl top nodes; Start-Sleep -Seconds 5 }"

# Start a new PowerShell window for pod status
Start-Process powershell -ArgumentList "-NoExit", "-Command", "while (`$true) { Clear-Host; Write-Host 'Pod Status:'; kubectl get pods -n ragbot -o wide; Start-Sleep -Seconds 5 }"

Write-Host "Monitoring windows have been opened. Close them when you're done monitoring." 