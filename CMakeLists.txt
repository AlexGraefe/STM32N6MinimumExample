cmake_minimum_required(VERSION 3.22)

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Compiler options
set(STM32_MCU_FLAGS  "-mcpu=cortex-m55 -mfpu=fpv5-d16 -mfloat-abi=hard -mcmse" )

#Linker options
set(STM32_LINKER_SCRIPT linker/STM32N657xx.ld)
set(STM32_LINKER_OPTION "-u _printf_float")

# Set the project name
set(CMAKE_PROJECT_NAME Appli)
project(${CMAKE_PROJECT_NAME})

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# System macros
set(MX_Defines_Syms 
	STM32N657xx
    $<$<CONFIG:Debug>:DEBUG>
    USE_FULL_ASSERT
    USE_FULL_LL_DRIVER
    VECT_TAB_SRAM
    LL_ATON_DUMP_DEBUG_API
    LL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6
    LL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL
    LL_ATON_RT_MODE=LL_ATON_RT_ASYNC
    LL_ATON_SW_FALLBACK
    LL_ATON_DBG_BUFFER_INFO_EXCLUDED=1
)
# System include paths
set(MX_Include_Dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Inc
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Inc
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32N6xx/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/CMSIS/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/CMSIS/DSP/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/BSP/STM32N6570-DK
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/Devices/STM32N6xx/
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Inc
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton
    ${CMAKE_CURRENT_SOURCE_DIR}/Model
)


# System application sources
set(MX_Application_Src
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/stm32n6xx_it.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/syscalls.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32N6xx/Source/Templates/gcc/startup_stm32n657xx.s
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Src/app_fuseprogramming.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Model/network.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Model/stai_network.c
)

# STM32 HAL/LL Drivers
set(STM32_Drivers_Src
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32N6xx/Source/Templates/system_stm32n6xx_fsbl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_bsec.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_gpio.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cortex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_exti.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_mdf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_sai.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_sai_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_xspi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rif.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cacheaxi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_uart.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_ramcfg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_xspi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/BSP/Components/mx66uw1g45g/mx66uw1g45g.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/BSP/Components/aps256xx/aps256xx.c
    # all files needed for neural network inference.
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/Devices/STM32N6xx/npu_cache.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/Devices/STM32N6xx/mcu_cache.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_rt_main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_runtime.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_cipher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_stai_internal.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_util.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_sw_float.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_sw_integer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_lib.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_lib_sw_operators.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_lib_sw_operators.c
)



# Link directories setup
set(MX_LINK_DIRS
${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Lib/GCC/ARMCortexM55
)
# Project libraries
set (MX_LINK_LIBS 
    STM32_Drivers
    ${TOOLCHAIN_LINK_LIBRARIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/stedgeai-lib/Lib/GCC/ARMCortexM55/NetworkRuntime1100_CM55_GCC.a
)

# Interface library for includes and symbols
add_library(stm32cubemx INTERFACE)
target_include_directories(stm32cubemx INTERFACE ${MX_Include_Dirs})
target_compile_definitions(stm32cubemx INTERFACE ${MX_Defines_Syms})

# Create STM32_Drivers static library
add_library(STM32_Drivers OBJECT)
target_sources(STM32_Drivers PRIVATE ${STM32_Drivers_Src})
target_link_libraries(STM32_Drivers PUBLIC stm32cubemx)

# Add STM32CubeMX generated application sources to the project
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${MX_Application_Src})

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${MX_LINK_DIRS})

# Add libraries to the project
target_link_libraries(${CMAKE_PROJECT_NAME} ${MX_LINK_LIBS})

# Add the map file to the list of files to be removed with 'clean' target
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_PROJECT_NAME}.map)

# In order to use CMake for cross-compiling
set(CMAKE_CXX_COMPILER_FORCED "true")

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined libraries
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Validate that STM32CubeMX code is compatible with C standard
if((CMAKE_C_STANDARD EQUAL 90) OR (CMAKE_C_STANDARD EQUAL 99))
    message(ERROR "Generated code requires C11 or higher")
endif()

# Post-build steps to sign binary
########################
# Specify the output ELF file
set(TARGET_ELF "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.elf")
# Command to generate .bin file
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY}
-O binary ${TARGET_ELF} ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.bin
    COMMENT "Generating .bin file"
)

# Define the paths to the STM32 Signing Tool and the input/output files
set(SIGNING_TOOL "STM32_SigningTool_CLI")
# Define the input and output files for Appli
set(Appli_INPUT "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.bin")
set(Appli_OUTPUT "${CMAKE_BINARY_DIR}/Appli-trusted.bin")

# Add a post-build step to sign the Appli binary
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo y > input.txt
    COMMAND "${SIGNING_TOOL}" -bin "${Appli_INPUT}" -nk -of 0x80000000 -t fsbl -o "${Appli_OUTPUT}" -hv 2.3 -dump "${Appli_OUTPUT}" < input.txt -align
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Signing Appli binary: ${Appli_INPUT} -> ${Appli_OUTPUT}"
)

# Convert the signed Appli .bin back to .elf
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -I binary -O elf32-littlearm -B arm --change-addresses=0x70100000 ${Appli_OUTPUT} ${CMAKE_BINARY_DIR}/Appli-trusted.elf
    COMMENT "Converting signed Appli .bin back to .elf in address [0x70100000]"
)