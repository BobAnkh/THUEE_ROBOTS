echo "xilinx" | sudo -S  systemctl stop serial-getty@ttyPS0.service
sleep 2 
echo "xilinx" | sudo -S  systemctl disable serial-getty@ttyPS0.service
