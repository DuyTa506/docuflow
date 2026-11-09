# First-logon setup of the library workstation (POC).
$ErrorActionPreference = "Continue"
$docuflow = "192.168.212.60"

# Static LAN address: this LAN has no DHCP server.
$nic = Get-NetAdapter | Where-Object Status -eq "Up" | Select-Object -First 1
Get-NetIPAddress -InterfaceIndex $nic.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
New-NetIPAddress -InterfaceIndex $nic.ifIndex -IPAddress 192.168.212.250 -PrefixLength 24 `
    -DefaultGateway 192.168.212.1
Set-DnsClientServerAddress -InterfaceIndex $nic.ifIndex -ServerAddresses 8.8.8.8
Rename-Computer -NewName "THUVIEN-PC" -Force
Start-Sleep 10
Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private

# WinRM so the DocuFlow host can finish the setup; reachable from DocuFlow only.
Enable-PSRemoting -Force -SkipNetworkProfileCheck
Set-Item WSMan:\localhost\Service\AllowUnencrypted $true
Set-Item WSMan:\localhost\Service\Auth\Basic $true
Get-NetFirewallRule -DisplayGroup "Windows Remote Management" | Disable-NetFirewallRule
New-NetFirewallRule -DisplayName "WinRM from DocuFlow" -Direction Inbound -Protocol TCP -LocalPort 5985 -RemoteAddress $docuflow -Action Allow
New-NetFirewallRule -DisplayName "DocuFlow callback" -Direction Inbound -Protocol TCP -LocalPort 9099 -RemoteAddress $docuflow -Action Allow
New-NetFirewallRule -DisplayName "SQL Server from DocuFlow" -Direction Inbound -Protocol TCP -LocalPort 1433 -RemoteAddress $docuflow -Action Allow
New-NetFirewallRule -DisplayName "ICMP" -Direction Inbound -Protocol ICMPv4 -Action Allow
netsh http add urlacl url=http://+:9099/ user=Everyone
"done" | Out-File C:\OEM\install.done
Restart-Computer -Force
