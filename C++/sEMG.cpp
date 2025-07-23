#include <ncnn/net.h>
#include <cnpy.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fmt/core.h>

int main() {
    // ģ�Ͳ�����Ȩ��·�����������Ŀ��Ŀ¼��
    std::string experiment_name = "E1";

    const std::string model_param = fmt::format(R"(D:\Project\sEMG\output_model\model_{}\{}.ncnn-opt.param)", experiment_name, experiment_name);
    const std::string model_bin = fmt::format(R"(D:\Project\sEMG\output_model\model_{}\{}.ncnn-opt.bin)", experiment_name, experiment_name);
    
    // const std::string model_param = fmt::format(R"(D:\Project\sEMG\output_model\model_{}\{}int8.param)", experiment_name, experiment_name);
    // const std::string model_bin = fmt::format(R"(D:\Project\sEMG\output_model\model_{}\{}int8.bin)", experiment_name, experiment_name);

    // ��������·�������㵼���ı���һ��
    const std::string npy_x = fmt::format(R"(D:/Project/sEMG/test_data_old\{}_X.npy)", experiment_name, experiment_name);
    const std::string npy_y = fmt::format(R"(D:/Project/sEMG/test_data_old\{}_y.npy)", experiment_name, experiment_name);

    std::cout << "[INFO] ��ʼ���ز�������..." << std::endl;
    // ��ȡ��������
    cnpy::NpyArray x_npy = cnpy::npy_load(npy_x);
    cnpy::NpyArray y_npy = cnpy::npy_load(npy_y);

    auto shape = x_npy.shape;
    size_t num = shape[0], ch = shape[1], h = shape[2], w = shape[3];
    std::cout << "[INFO] ����������״: ������=" << num << ", ͨ����=" << ch << ", �߶�=" << h << ", ���=" << w << std::endl;

    float* x_data = x_npy.data<float>();
    int64_t* y_data = y_npy.data<int64_t>();

    std::cout << "[INFO] ��ʼ����ģ��..." << std::endl;
    ncnn::Net net;
    std::cout << model_param.c_str() << std::endl;
    if (net.load_param(model_param.c_str()) != 0) 
    {
        return -1;
    }
    if (net.load_model(model_bin.c_str()) != 0) 
    {
        return -1;
    }
    std::cout << "[INFO] ģ�ͼ��سɹ�" << std::endl;

    int correct = 0;
    for (size_t i = 0; i < num; ++i) {
        if (i % 50 == 0) {
            std::cout << "[INFO] ������������ " << i << "/" << num << std::endl;
        }

        const float* sample = x_data + i * ch * h * w;

        int w_int = static_cast<int>(w);
        int h_int = static_cast<int>(h);
        int ch_int = static_cast<int>(ch);
        ncnn::Mat in(w_int, h_int, ch_int);

        // �����������
        for (int c = 0; c < ch_int; ++c) {
            for (int y = 0; y < h_int; ++y) {
                for (int x = 0; x < w_int; ++x) {
                    in.channel(c).row(y)[x] = sample[c * h * w + y * w + x];
                }
            }
        }

        // ģ������
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out;
        ex.extract("out0", out);

        // �������ʶ�Ӧ�����
        int pred_label = 0;
        float max_score = out[0];
        for (int j = 1; j < out.w; ++j) {
            if (out[j] > max_score) {
                max_score = out[j];
                pred_label = j;
            }
        }

        int true_label = static_cast<int>(y_data[i]);

        std::cout << "[DEBUG] ���� " << i << ": Ԥ���ǩ = " << pred_label
                  << ", ��ʵ��ǩ = " << true_label
                  << ", ���Ŷ� = " << max_score << std::endl;

        if (pred_label == true_label)
            ++correct;
    }

    float acc = static_cast<float>(correct) / num;
    std::cout << "[RESULT] ��������: " << num << ", ��ȷ��: " << correct
              << ", ׼ȷ��: " << acc * 100 << "%" << std::endl;
    std::cout << experiment_name << std::endl;

    return 0;
}
