# OBS Virtual Camera Status Helper (No Admin Needed)
# Minimal ASCII-only script to avoid parser issues on older PowerShell

Write-Host "=== OBS Virtual Camera Status Helper ==="

# 1) Check if OBS Studio process is running
$obs = Get-Process -Name obs64 -ErrorAction SilentlyContinue
if ($null -ne $obs) {
    Write-Host ("OBS Studio is running (PID: {0})" -f $obs.Id)
} else {
    Write-Host "OBS Studio is NOT running"
    Write-Host "Tip: Open OBS, go to Tools > Virtual Camera, then click Start"
}

# 2) List PnP camera devices (if cmdlet exists)
Write-Host ""
Write-Host "PnP camera devices (if available):"
if (Get-Command Get-PnpDevice -ErrorAction SilentlyContinue) {
    Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue | Select-Object FriendlyName, InstanceId | Format-Table -AutoSize
} else {
    Write-Host "Get-PnpDevice not available on this PowerShell version"
}

# 3) List WMI camera-like devices
Write-Host ""
Write-Host "WMI video-related devices:"
Get-WmiObject -Class Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'camera' -or $_.Name -match 'webcam' -or $_.Name -match 'video' } | Select-Object Name, DeviceID | Format-Table -AutoSize

# 4) Show typical OBS Virtual Camera names to look for
Write-Host ""
Write-Host "Typical OBS Virtual Camera names to look for:"
Write-Host "  - OBS Virtual Camera"
Write-Host "  - OBS-Camera"
Write-Host "  - OBS Virtual Source"

Write-Host ""
Write-Host "If the OBS Virtual Camera is not appearing:"
Write-Host "  1) In OBS: Tools > Virtual Camera > Start (and enable AutoStart)"
Write-Host "  2) Fully quit and restart OBS Studio"
Write-Host "  3) Reboot Windows if it still does not appear"
Write-Host "  4) Reinstall OBS and include the Virtual Camera component"

Write-Host ""
Write-Host "=== Helper Complete ==="
