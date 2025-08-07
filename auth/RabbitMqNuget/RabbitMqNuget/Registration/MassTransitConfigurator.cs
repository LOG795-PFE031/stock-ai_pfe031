using MassTransit;
using RabbitMQ.Client;

namespace RabbitMqNuget.Registration;

public sealed class MassTransitConfigurator
{
    public IEnumerable<Action<IBusRegistrationConfigurator, IBusRegistrationContext, IRabbitMqBusFactoryConfigurator>> ConfigureMessages => _configureMessages;

    public IReadOnlyDictionary<Type, string> ExchangeNamesForMessageTypes => _exchangeNamesForMessageTypes;

    private readonly HashSet<Type> _messageTypes = [];

    private readonly List<Action<IBusRegistrationConfigurator, IBusRegistrationContext, IRabbitMqBusFactoryConfigurator>> _configureMessages = [];

    private readonly Dictionary<Type, string> _exchangeNamesForMessageTypes = new();

    public MassTransitConfigurator AddConsumer<TMessage, TConsumer>(string exchangeName)
        where TMessage : class
        where TConsumer : class, IConsumer<TMessage>, new() => AddConsumer<TMessage, TConsumer>(exchangeName, _ => new TConsumer());

    public MassTransitConfigurator AddConsumer<TMessage, TConsumer>(string exchangeName, Func<IServiceProvider, TConsumer> consumerFactory) 
        where TMessage : class
        where TConsumer : class, IConsumer<TMessage>
    {
        _configureMessages.Add((_, ctx, configurator) =>
        {
            RegisterMessageType<TMessage>(configurator);

            configurator.ReceiveEndpoint($"{typeof(TMessage).Name}.queue", endpoint =>
            {
                endpoint.ConfigureConsumeTopology = false;

                endpoint.Bind(exchangeName, binding =>
                {
                    binding.ExchangeType = ExchangeType.Fanout;
                });

                endpoint.Consumer(() => consumerFactory(ctx));
            });
        });

        return this;
    }

    public MassTransitConfigurator AddPublisher<TMessage>(string exchangeName) 
        where TMessage : class
    {
        _configureMessages.Add((_, _, configurator) =>
        {
            RegisterMessageType<TMessage>(configurator);

            configurator.Publish<TMessage>(cfg =>
            {
                cfg.Exclude = true;
                cfg.ExchangeType = ExchangeType.Fanout;
            });
        });

        _exchangeNamesForMessageTypes.Add(typeof(TMessage), exchangeName);

        return this;
    }


    private void RegisterMessageType<TMessage>(IRabbitMqBusFactoryConfigurator configurator) 
        where TMessage : class
    {
        if (_messageTypes.Contains(typeof(TMessage)))
        {
            return;
        }

        configurator.UseRawJsonDeserializer(RawSerializerOptions.All, isDefault: true);

        configurator.Message<TMessage>(topologyConfigurator =>
        {              
            topologyConfigurator.SetEntityNameFormatter(new MessageEntityNameFormatter<TMessage>(new MessageNameFormatter()));
        });

        _messageTypes.Add(typeof(TMessage));
    }
}