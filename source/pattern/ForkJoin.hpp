#pragma once

#include "../Commons.hpp"
#include "../interfaces/PatternInterface.hpp"
#include "../interfaces/Executor.hpp"

template <typename T_input, typename T_intermediate1, typename T_intermediate2, typename T_output>
class ForkJoin : public PatternInterface<T_input, T_output>
{
private:
    Executor<T_input, T_intermediate1> executor_fork1;
    Executor<T_input, T_intermediate2> executor_fork2;
    Executor<T_intermediate1, T_output> executor_join1;
    Executor<T_intermediate2, T_output> executor_join2;
    Executor<std::tuple<T_intermediate1 &, T_intermediate2 &>, bool> executor_comparator;

    ForkJoin(PatIntPtr<T_input, T_intermediate1> &fork1,
             PatIntPtr<T_input, T_intermediate2> &fork2,
             PatIntPtr<T_intermediate1, T_output> &join1,
             PatIntPtr<T_intermediate2, T_output> &join2,
             PatIntPtr<std::tuple<T_intermediate1 &, T_intermediate2 &>, bool> &comparator) : executor_fork1(fork1),
                                                                                            executor_fork2(fork2),
                                                                                            executor_join1(join1),
                                                                                            executor_join2(join2),
                                                                                            executor_comparator(comparator)
    {
    }

protected:
    void InternallyCompute(std::future<T_input> future, std::promise<T_output> promise) override
    {
        auto input{future.get()};
        auto input_copy{input};

        const auto wrapIntoFuture = [](auto &&v) {
            std::promise<T_input> p;
            p.set_value(std::move(v));
            return p.get_future();
        };

        auto input_future_fork1{wrapIntoFuture(std::move(input))};
        auto input_future_fork2{wrapIntoFuture(std::move(input_copy))};

        auto intermediate_future_fork1{executor_fork1.Compute(std::move(input_future_fork1))};
        auto intermediate_future_fork2{executor_fork2.Compute(std::move(input_future_fork2))};

        auto intermediate_result_fork1{intermediate_future_fork1.get()};
        auto intermediate_result_fork2{intermediate_future_fork2.get()};

        std::tuple<T_intermediate1 &, T_intermediate2&> tuple(intermediate_result_fork1,intermediate_result_fork2);

        if (executor_comparator.Compute(std::move(tuple)))
        {
            auto f(wrapIntoFuture(std::move(intermediate_result_fork1)));
            executor_join1.Compute(std::move(f), std::move(promise));
        }
        else
        {
            auto f(wrapIntoFuture(std::move(intermediate_result_fork2)));
            executor_join2.Compute(std::move(f), std::move(promise));
        }
    }

public:
    static PatIntPtr<T_input, T_output> create(PatIntPtr<T_input, T_intermediate1> fork1,
                                               PatIntPtr<T_input, T_intermediate2> fork2,
                                               PatIntPtr<T_intermediate1, T_output> join1,
                                               PatIntPtr<T_intermediate2, T_output> join2,
                                               PatIntPtr<std::tuple<T_intermediate1 &, T_intermediate2&>, bool> comparator)
    {
        auto fj = new ForkJoin(fork1, fork2, join1, join2, comparator);
        auto s_ptr = std::shared_ptr<PatternInterface<T_input, T_output>>(fj);
        return s_ptr;
    }

    ForkJoin(const ForkJoin &) = delete;
    ForkJoin(ForkJoin &&) = delete;

    ForkJoin &operator=(const ForkJoin &) = delete;
    ForkJoin &operator=(ForkJoin &&) = delete;

    PatIntPtr<T_input, T_output> create_copy() override
    {
        this->assertNoInit();

        auto copied_version = create(executor_fork1.GetTask(), executor_fork2.GetTask(), executor_join1.GetTask(), executor_join2.GetTask(), executor_comparator.GetTask());
        return copied_version;
    }

    void Init() override
    {
        if (!this->initialized)
        {
            this->dying = false;

            executor_fork1.Init();
            executor_fork2.Init();
            executor_join1.Init();
            executor_join2.Init();
            executor_comparator.Init();

            this->initialized = true;
        }
    }

    void Dispose() override
    {
        if (this->initialized)
        {
            this->dying = true;

            executor_fork1.Dispose();
            executor_fork2.Dispose();
            executor_join1.Dispose();
            executor_join2.Dispose();
            executor_comparator.Dispose();

            this->initialized = false;
        }
    }

    size_t ThreadCount() const noexcept override
    {
        return executor_fork1.ThreadCount() + executor_fork2.ThreadCount() + executor_join1.ThreadCount() + executor_join2.ThreadCount() + executor_comparator.ThreadCount();
    }

    std::string Name() const override
    {
        return std::string("ForkJoin(") + executor_fork1.Name() + std::string(",") + executor_fork2.Name() + std::string(",") + executor_join1.Name() + std::string(",") + executor_join2.Name() + std::string(",") + executor_comparator.Name() + std::string(")");
    }

    ~ForkJoin()
    {
        ForkJoin<T_input, T_intermediate1, T_intermediate2, T_output>::Dispose();
    }
};
