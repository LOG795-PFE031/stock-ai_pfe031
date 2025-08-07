using System.Collections.Concurrent;

namespace AuthService.HostedServices;

public sealed class ServiceReady
{
    public static ServiceReady Instance { get; } = new();

    public TaskCompletionSource<bool> IsReady { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

    private readonly ConcurrentDictionary<Type, bool> s_notifiedReadyRequiredTypes = new ();

    internal ServiceReady()
    {
        s_notifiedReadyRequiredTypes.TryAdd(typeof(AddDefaultDbRecords), false);
    }

    public void Ready<TReady>()
    {
        s_notifiedReadyRequiredTypes[typeof(TReady)] = true;

        if (s_notifiedReadyRequiredTypes.Values.All(x => x))
        {
            IsReady.SetResult(true);
        }
    }
}