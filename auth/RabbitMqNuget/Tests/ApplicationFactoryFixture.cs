using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.VisualStudio.TestPlatform.TestHost;
using RabbitMqNuget.Registration;
using RabbitMqNuget.Testing;
using Tests.Rabbitmq.Consumers;
using Tests.Rabbitmq.Messages.Impl;

namespace Tests;

public sealed class ApplicationFactoryFixture : WebApplicationFactory<Program>, IAsyncLifetime
{
    private readonly RabbitmqTestContainer _rabbitMq = new ();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        base.ConfigureWebHost(builder);

        builder.ConfigureAppConfiguration((_, config) =>
        {
            var integrationConfig = new Dictionary<string, string>
            {
                ["ConnectionStrings:Rabbitmq"] = _rabbitMq.Container.GetConnectionString(),
            };
            
            config.AddInMemoryCollection(integrationConfig!);
        });

        builder.ConfigureServices(services =>
        {
            // Remove any existing MassTransit registrations to prevent duplicate configuration.
            var massTransitDescriptors = services
                .Where(s => s.ServiceType?.Namespace?.Contains("MassTransit") == true)
                .ToList();

            foreach (var descriptor in massTransitDescriptors)
            {
                services.Remove(descriptor);
            }

            services.RegisterMassTransit(
                _rabbitMq.Container.GetConnectionString(),
                new MassTransitConfigurator()
                    .AddPublisher<TestMessage>("test-exchange")
                    .AddConsumer<TestMessage, ConsumerDecorator<TestMessage, TestMessageConsumer>>("test-exchange",
                        _ => new(new())));
        });
    }

    public async Task InitializeAsync()
    {
        await _rabbitMq.InitializeAsync();
    }

    public new async Task DisposeAsync()
    {
        await base.DisposeAsync();

        await _rabbitMq.DisposeAsync();
    }
}