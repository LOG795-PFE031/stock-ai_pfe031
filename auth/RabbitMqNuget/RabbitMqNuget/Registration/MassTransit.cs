using MassTransit;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using RabbitMqNuget.Middlewares;
using RabbitMqNuget.Services.Impl;
using RabbitMqNuget.Services;

namespace RabbitMqNuget.Registration
{
    public static class MassTransit
    {
        public static IReadOnlyDictionary<Type, string> ExchangeNamesForMessageTypes { get; private set; } = new Dictionary<Type, string>();

        public static void RegisterMassTransit(this IServiceCollection services, string connectionString, MassTransitConfigurator massTransitConfigurator)
        {
            services.AddScoped<ITransactionInfo, TransactionInfo>();

            services.AddScoped(typeof(IMessagePublisher<>), typeof(MessagePublisher<>));

            services.Configure<RabbitMqOptions>(options =>
            {
                options.Rabbitmq = connectionString;
            });

            services.AddMassTransit(busRegistrationConfigurator =>
            {
                busRegistrationConfigurator.UsingRabbitMq((busRegistrationContext, rabbitMqBusFactoryConfigurator) =>
                {
                    rabbitMqBusFactoryConfigurator.Host(connectionString);

                    rabbitMqBusFactoryConfigurator.UseMessageRetry(r => r.Immediate(5));

                    foreach (var configureMessage in massTransitConfigurator.ConfigureMessages)
                    {
                        configureMessage(busRegistrationConfigurator, busRegistrationContext, rabbitMqBusFactoryConfigurator);
                    }
                });
            });

            ExchangeNamesForMessageTypes = massTransitConfigurator.ExchangeNamesForMessageTypes;
        }

        public static void AddTransactionMiddleware(this IApplicationBuilder app)
        {
            app.UseMiddleware<TransactionMiddleware>();
        }
    }
}
