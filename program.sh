#!/bin/bash
STM32_Programmer_CLI -c port=SWD  -d ./build/Debug/Appli-trusted.bin 0x70100000 -el /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr 
if [ $? -ne 0 ]; then
    echo ""
    echo -e "\033[0;31mError: Failed to program Application. Please see error messages above.\033[0m" >&2
    exit 1
fi


arm-none-eabi-objcopy -I binary examples/st_ai_output/network_atonbuf.xSPI2.raw --change-addresses 0x70380000 -O ihex examples/st_ai_output/network_data.hex
if [ $? -ne 0 ]; then
    echo ""
    echo -e "\033[0;31mError: Failed to convert from bin to hex. Please see error messages above.\033[0m" >&2
    exit 1
fi

STM32_Programmer_CLI -c port=SWD  -d examples/st_ai_output/network_data.hex -el /usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr 
if [ $? -ne 0 ]; then
    echo ""
    echo -e "\033[0;31mError: Failed to program Application. Please see error messages above.\033[0m" >&2
    exit 1
fi

echo ""
echo -e "\033[0;32mSuccessfully wrote FSBL and application binaries into memory!\033[0m"
