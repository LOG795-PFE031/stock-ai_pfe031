using FluentAssertions;
using RabbitMqNuget.Testing;
using Tests.Rabbitmq.Messages.Impl;

namespace Tests.Rabbitmq;

[Collection(nameof(TestCollections.Default))]
public class ConnectionTests
{
    private readonly ApplicationFactoryFixture _applicationFactoryFixture;

    public ConnectionTests(ApplicationFactoryFixture applicationFactoryFixture)
    {
        _applicationFactoryFixture = applicationFactoryFixture;
    }

    [Fact]
    public async Task WithTestExchangeEndpoint_SendMessage_ShouldReturnCorrectMessage()
    {
        const string message = "Hello, World!";

        var responseMessage = await _applicationFactoryFixture.Services.WithMessagePublished(new TestMessage()
        {
            Message = message
        });

        responseMessage.Should().NotBeNull();
        responseMessage.Message.Should().Be(message);
    }
}