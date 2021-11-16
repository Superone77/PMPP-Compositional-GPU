#pragma once

#include "../Commons.hpp"

#include <optional>
#include <functional>
#include <set>
#include <atomic>
#include <shared_mutex>

class LoadBalancer;

struct LoadBalancingBase;

struct LoadBalancingBase
{
    virtual bool doTask() { return false; };
    std::shared_ptr<LoadBalancer> load_balancer;
    std::weak_ptr<LoadBalancingBase> self;
};

class LoadBalancer
{
    struct LoadBalancerSetComparator
    {
        bool operator()(std::weak_ptr<LoadBalancingBase> a, std::weak_ptr<LoadBalancingBase> b) const noexcept
        {
            auto s1 = a.lock();
            auto s2 = b.lock();
            return s1 < s2;
        }
    };

    std::set<std::weak_ptr<LoadBalancingBase>, LoadBalancerSetComparator> load_balancing_patterns{};
    mutable std::shared_mutex m;

public:
    static void registerLoadBalancer(std::weak_ptr<LoadBalancingBase> pattern_ptr, std::shared_ptr<LoadBalancer> load_balancer)
    {
        std::unique_lock<std::shared_mutex> lg(load_balancer->m);
        if (auto s_ptr = pattern_ptr.lock())
            s_ptr->load_balancer = load_balancer;
        load_balancer->load_balancing_patterns.insert(pattern_ptr);
    }

    static void deregisterLoadBalancer(std::weak_ptr<LoadBalancingBase> pattern_ptr, std::shared_ptr<LoadBalancer> load_balancer)
    {
        std::unique_lock<std::shared_mutex> lg(load_balancer->m);
        load_balancer->load_balancing_patterns.erase(pattern_ptr);
    }

    static std::shared_ptr<LoadBalancer> create() { return std::make_shared<LoadBalancer>(); }

    bool provideWork() const
    {
        std::shared_lock<std::shared_mutex> sl(m);
        for (auto &v : load_balancing_patterns)
        {
            if (auto s_ptr = v.lock())
            {
                if (s_ptr->doTask())
                    return true;
            }
        }
        return false;
    }
};
