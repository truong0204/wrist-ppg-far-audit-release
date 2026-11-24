idf_component_register(SRCS "main.cc" "h32_core.c" "model_data.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_driver_uart esp_timer)
