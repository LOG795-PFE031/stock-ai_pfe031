using MassTransit;
using Tests.Rabbitmq.Messages.Impl;

namespace Tests.Rabbitmq.Consumers;

public sealed class TestMessageConsumer : IConsumer<TestMessage>
{
    public Task Consume(ConsumeContext<TestMessage> context)
    {
        return Task.CompletedTask;
    }
}