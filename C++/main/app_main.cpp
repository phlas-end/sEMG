#include "dl_model_base.hpp"
#include <cmath>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "lwip/sockets.h"
#include <cstring>
#include "esp_timer.h"   // 时间统计
#include "esp_system.h"  // 堆内存信息
#include "esp_heap_caps.h"

extern const uint8_t model_espdl[] asm("_binary_sEMG_espdl_start");

#define PORT 3333
static const char* TAG = "app";

dl::Model *model;

// ================= TCP Server Task =================
void tcp_server_task(void *)
{
    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket");
        return;
    }

    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    bind(listen_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    listen(listen_sock, 1);
    ESP_LOGI(TAG, "TCP server listening on port %d", PORT);

    const int input_size = 1*200*16;        // 输入元素数
    const int input_bytes = input_size * sizeof(float);

    // ===== 全局统计变量 =====
    int total_runs = 0;
    double total_model_time = 0;
    double total_round_time = 0;
    double max_model_time = 0;
    double min_model_time = 1e9;
    size_t total_used_mem = 0;

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_sock = accept(listen_sock, (struct sockaddr*)&client_addr, &addr_len);
        if (client_sock < 0) continue;

        ESP_LOGI(TAG, "Client connected");

        // ===== 整个流程开始时间 =====
        int64_t round_start = esp_timer_get_time();

        // ===== 循环接收完整输入 =====
        std::vector<float> input_buf(input_size);
        uint8_t *ptr = (uint8_t*)input_buf.data();
        size_t received = 0;
        while (received < input_bytes) {
            int r = recv(client_sock, ptr + received, input_bytes - received, 0);
            if (r <= 0) {
                ESP_LOGE(TAG, "Connection closed or error during recv");
                break;
            }
            received += r;
        }

        if (received != input_bytes) {
            ESP_LOGE(TAG, "Recv size mismatch: got %d bytes, expected %d bytes", (int)received, input_bytes);
            close(client_sock);
            continue;
        }

        // ===== 量化输入 =====
        auto inputs = model->get_inputs();
        auto input_tensor = inputs.begin()->second;
        int8_t *in_ptr = (int8_t*)input_tensor->data;
        for (int i = 0; i < input_size; i++) {
            in_ptr[i] = dl::quantize<int8_t>(input_buf[i], DL_RESCALE(input_tensor->exponent));
        }

        // ===== 模型推理计时 =====
        int64_t model_start = esp_timer_get_time();
        model->run();
        int64_t model_end = esp_timer_get_time();

        double model_time_ms = (model_end - model_start) / 1000.0;

        // ===== 输出 52 类，取 argmax =====
        auto outputs = model->get_outputs();
        auto output_tensor = outputs.begin()->second;
        int output_size = 5;
        int8_t *out_ptr = (int8_t*)output_tensor->data;

        int max_index = 0;
        float max_value = dl::dequantize(out_ptr[0], DL_SCALE(output_tensor->exponent));
        for (int i = 1; i < output_size; i++) {
            float v = dl::dequantize(out_ptr[i], DL_SCALE(output_tensor->exponent));
            if (v > max_value) {
                max_value = v;
                max_index = i;
            }
        }

        ESP_LOGI(TAG, "Predicted class: %d, value = %f", max_index, max_value);

        // ===== 返回类别索引 =====
        send(client_sock, &max_index, sizeof(max_index), 0);
        close(client_sock);

        // ===== 整个流程耗时 =====
        int64_t round_end = esp_timer_get_time();
        double round_time_ms = (round_end - round_start) / 1000.0;

        // ===== 内存统计 =====
        size_t free_heap = esp_get_free_heap_size();
        size_t used_mem = heap_caps_get_total_size(MALLOC_CAP_DEFAULT) - free_heap;
        total_used_mem += used_mem;

        // ===== 更新统计指标 =====
        total_runs++;
        total_model_time += model_time_ms;
        total_round_time += round_time_ms;
        if (model_time_ms > max_model_time) max_model_time = model_time_ms;
        if (model_time_ms < min_model_time) min_model_time = model_time_ms;

        // ===== 打印统计 =====
        ESP_LOGI(TAG, "单次模型耗时: %.3f ms | 单次请求总耗时: %.3f ms", model_time_ms, round_time_ms);
        ESP_LOGI(TAG, "平均模型耗时: %.3f ms | 平均请求耗时: %.3f ms | 最快: %.3f ms | 最慢: %.3f ms (共%d次)",
                 total_model_time / total_runs,
                 total_round_time / total_runs,
                 min_model_time,
                 max_model_time,
                 total_runs);

        ESP_LOGI(TAG, "内存: 当前可用 %d bytes | 历史最小可用 %d bytes | 平均已用 %d bytes",
                 free_heap,
                 esp_get_minimum_free_heap_size(),
                 (int)(total_used_mem / total_runs));
    }
}


// ================= Wi-Fi 事件回调 =================
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "STA start, connecting...");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected, reconnecting...");
        esp_wifi_connect();
    }
}

// ================= Wi-Fi 直连 STA =================
void wifi_init_sta()
{
    // Init NVS
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件回调
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    // 配置 SSID / 密码
    wifi_config_t wifi_config = {};
    strcpy((char *)wifi_config.sta.ssid, "HUAWEI-209");      // 修改为你的 WiFi 名
    strcpy((char *)wifi_config.sta.password, "woshidashabi"); // 修改为 WiFi 密码

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi STA init finished. Connecting to SSID:%s", wifi_config.sta.ssid);
}

// ================= app_main =================
extern "C" void app_main(void)
{
    // Load model
    model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    // Start Wi-Fi direct connect
    wifi_init_sta();

    // Start TCP server
    xTaskCreate(tcp_server_task, "tcp_server", 8192, NULL, 5, NULL);
}
