#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template<typename T>
class sync_queue
{
   public:
    T pop()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty())
        {
            cond_.wait(mlock);
        }
        auto val = queue_.front();
        queue_.pop();
        mlock.unlock();
        cond_.notify_one();
        return val;
    }

    void pop(T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty())
        {
            cond_.wait(mlock);
        }
        item = queue_.front();
        queue_.pop();
        mlock.unlock();
        cond_.notify_one();
    }

    void push(const T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push(item);
        mlock.unlock();
        cond_.notify_one();
    }
    unsigned int size()
    {
        return queue_.size();
    }
    sync_queue() = default;
    sync_queue(const sync_queue&) = delete;             // disable copying
    sync_queue& operator=(const sync_queue&) = delete;  // disable assignment

   private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};