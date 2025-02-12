//==============================================================
// Copyright (C) 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This is a simple DPC++ program that illustrates how to force work-items to be assigned to unique threads.
// The kernel is designed to process given data in parallel with making sure that every work-item
// is processed in separate thread.

#include <sycl/sycl.hpp>
#include <iostream>
#include <thread>
#include <xtimec.h>
#include <string>
#include <sstream>

#define CPU 1
#define GPU 2
#define OFFLOAD_TYPE CPU

int stallThread(int item);

extern SYCL_EXTERNAL _CRTIMP2_PURE _Thrd_id_t __cdecl _Thrd_id();
extern SYCL_EXTERNAL _CRTIMP2_PURE long long __cdecl _Query_perf_frequency();
extern SYCL_EXTERNAL _CRTIMP2_PURE long long __cdecl _Query_perf_counter();
extern SYCL_EXTERNAL _CRTIMP2_PURE long long __cdecl _Xtime_get_ticks();

//extern SYCL_EXTERNAL _CRTIMP2_PURE void __cdecl _Thrd_sleep(const xtime*); // VS2019

extern SYCL_EXTERNAL void __stdcall _Thrd_sleep_for(unsigned long /*ms*/) noexcept; // VS2022
int main()
{

#if OFFLOAD_TYPE == CPU
    constexpr size_t LENGTH = 3;
    // Length of the work-group.
    constexpr int threadCount = LENGTH;

    // Variables input array.
    int input[LENGTH] = { 12, 111, 3 };

    // Thread ids output array.
    std::thread::id threadIds[LENGTH];

    // Variable values output array.
    int variableValues[LENGTH];

    // Create queue with CPU selector.
    sycl::cpu_selector selector;
#elif OFFLOAD_TYPE == GPU
    constexpr size_t LENGTH = 12;
    // Length of the work-group.
    constexpr int threadCount = 2;

    // Variables input array.
    int input[threadCount];

    // Initialize input.
    for (unsigned int i = 0; i < threadCount; i++)
        input[i] = i + 100;

    // Thread ids output array.
    int threadIds[threadCount];
    // Variable values output array.
    int variableValues[threadCount];

    // Create queue with GPU selector.
    sycl::gpu_selector selector;
#endif // OFFLOAD_TYPE == CPU

    try {
        sycl::queue q(selector);
        std::stringstream offloadMessageStream;
        offloadMessageStream << "[SYCL] Using device: ["
            << q.get_device().get_info<sycl::info::device::name>()
            << "] with backend: ["
            << q.get_backend()
            << "]"
            << std::endl;
        auto offloadMessage = offloadMessageStream.str();
        std::cout << offloadMessage << std::endl;
        sycl::range data_range{ threadCount }; // OFFLOAD_MESSAGE_BREAKPOINT
        sycl::buffer buffer_in{ input, data_range };
        sycl::buffer buffer_threadIds{ threadIds, data_range };
        sycl::buffer buffer_variableValues{ variableValues, data_range };

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::time_point::min();
        q.submit([&](sycl::handler& cgh) {
            // Memory accessors for buffers.
            sycl::accessor in(buffer_in, cgh, sycl::read_only);
            sycl::accessor out_threadIds(buffer_threadIds, cgh, sycl::write_only);
            sycl::accessor out_variableValues(buffer_variableValues, cgh, sycl::write_only);

            // Timer start.
            start = std::chrono::high_resolution_clock::now();

            // NOTE: We are assuming that the iterator starts from 0 and increments by 1
            // kernel start.
            cgh.parallel_for(sycl::range{ LENGTH }, [=](sycl::item <1> it) {
#if OFFLOAD_TYPE == CPU
                std::thread::id this_id = std::this_thread::get_id(); // Get current thread id.
                out_threadIds[it.get_id()] = this_id; // Assign thread id to the out_threadIds acessor.
                int currentWorkItem = in[it.get_id()]; // Get current work item value.
                out_variableValues[it.get_id()] = stallThread(in[it.get_id()]); // BREAKPOINT_1
#elif OFFLOAD_TYPE == GPU
                int threadId = 0;
                int index = it.get_id() / 8; // Devide by 8 to get correct index value.
                int currentWorkItem = in[index]; // Get current work item value.
                out_variableValues[index] = currentWorkItem;
                out_threadIds[index] = threadId; // BREAKPOINT_2
#endif // OFFLOAD_TYPE == CPU
                });
            }); // kernel end

        //Wait for queue end execution.
        q.wait_and_throw();

        // Stop the timer, calculate execution time and print it to the console.
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "\nTime taken by kernel: " << duration.count() << " milliseconds\n" << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cout << "fail; synchronous exception occurred: " << e.what() << "\n";
        return -1;
    }

    // Print the thread ids and corresponding variable values.
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < threadCount; i++) {
        std::cout << "Current thread id: " << threadIds[i] << " Value: " << variableValues[i] << "\n";
    }

    // Verify the variableValues output array.
    for (int i = 0; i < threadCount; i++) {
        if (variableValues[i] != input[i]) {
            std::cout << "\nfail; element " << i << " is " << variableValues[i] << std::endl;
            return -1;
        }
    }
    std::cout << "\nsuccess; result is correct.\n";
    return 0;
}

// NOTE: Below function is used to stall thread to make sure that uniqe thread is used for each work item.
int stallThread(int item) {
    //for (int64_t i = 0; i < 9000000000; i++) { }
    _Thrd_sleep_for(5000);
    return item;
}
