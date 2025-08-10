namespace AuthService.Configurations;

public sealed class PasswordSettings
{
    public required int RequiredLength { get; set; }
    public required bool RequireDigit { get; set; }
    public required bool RequireLowercase { get; set; }
    public required bool RequireUppercase { get; set; }
    public required bool RequireNonAlphanumeric { get; set; }
    public required int PreventPasswordReuseCount { get; set; }
    public required int MaxPasswordAge { get; set; }
}