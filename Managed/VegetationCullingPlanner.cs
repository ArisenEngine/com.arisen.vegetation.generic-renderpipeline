using ArisenEngine.Resources.Serialization;
using ArisenEngine.Vegetation;
using ArisenEngine.Vegetation.Assets;
using System.Numerics;

namespace ArisenEngine.Vegetation.GenericRenderPipeline;

internal enum VegetationCullingProjection
{
    Perspective = 0,
    Orthographic = 1
}

internal readonly record struct VegetationCullingView(
    WorldPosition CameraWorldPosition,
    WorldPosition RenderOrigin,
    Matrix4x4 OriginRelativeViewProjection,
    VegetationCullingProjection Projection,
    double VerticalFieldOfViewRadians,
    double OrthographicHeight,
    int ViewportHeight)
{
    public bool IsValid =>
        CameraWorldPosition.IsFinite &&
        RenderOrigin.IsFinite &&
        ViewportHeight > 0 &&
        IsFinite(OriginRelativeViewProjection) &&
        (Projection == VegetationCullingProjection.Perspective
            ? double.IsFinite(VerticalFieldOfViewRadians) &&
              VerticalFieldOfViewRadians > 0.0 &&
              VerticalFieldOfViewRadians < Math.PI
            : Projection == VegetationCullingProjection.Orthographic &&
              double.IsFinite(OrthographicHeight) &&
              OrthographicHeight > 0.0);

    public double ProjectionScale => Projection == VegetationCullingProjection.Perspective
        ? ViewportHeight / (2.0 * Math.Tan(VerticalFieldOfViewRadians * 0.5))
        : ViewportHeight / OrthographicHeight;

    private static bool IsFinite(in Matrix4x4 matrix) =>
        float.IsFinite(matrix.M11) && float.IsFinite(matrix.M12) &&
        float.IsFinite(matrix.M13) && float.IsFinite(matrix.M14) &&
        float.IsFinite(matrix.M21) && float.IsFinite(matrix.M22) &&
        float.IsFinite(matrix.M23) && float.IsFinite(matrix.M24) &&
        float.IsFinite(matrix.M31) && float.IsFinite(matrix.M32) &&
        float.IsFinite(matrix.M33) && float.IsFinite(matrix.M34) &&
        float.IsFinite(matrix.M41) && float.IsFinite(matrix.M42) &&
        float.IsFinite(matrix.M43) && float.IsFinite(matrix.M44);
}

internal readonly record struct VegetationCullingSettings(
    double HysteresisFraction,
    double DensityMultiplier,
    int MaximumInstanceCount,
    int MaximumBatchCount,
    double MaximumDistance,
    bool EnableFrustumCulling,
    bool EnableDistanceCulling)
{
    public static VegetationCullingSettings Default { get; } = new(
        HysteresisFraction: 0.15,
        DensityMultiplier: 1.0,
        MaximumInstanceCount: 1_048_576,
        MaximumBatchCount: 8_192,
        MaximumDistance: 4_096.0,
        EnableFrustumCulling: true,
        EnableDistanceCulling: true);

    public bool IsValid =>
        double.IsFinite(HysteresisFraction) &&
        HysteresisFraction >= 0.0 &&
        HysteresisFraction < 0.5 &&
        double.IsFinite(DensityMultiplier) &&
        DensityMultiplier >= 0.0 &&
        DensityMultiplier <= 1.0 &&
        MaximumInstanceCount > 0 &&
        MaximumBatchCount > 0 &&
        double.IsFinite(MaximumDistance) &&
        MaximumDistance > 0.0;
}

internal readonly struct VegetationClusterCullingInput
{
    public VegetationClusterCullingInput(
        in VegetationClusterComponent component,
        VegetationResidentClusterData resident,
        in VegetationPreparedClusterView prepared)
    {
        Component = component;
        Resident = resident ?? throw new ArgumentNullException(nameof(resident));
        Prepared = prepared;
    }

    public VegetationClusterComponent Component { get; }
    public VegetationResidentClusterData Resident { get; }
    public VegetationPreparedClusterView Prepared { get; }

    public bool IsValid =>
        Resident != null &&
        Prepared.IsValid &&
        Prepared.Generation == Resident.Generation &&
        Component.ClusterGuid == Resident.Guid &&
        Component.ClusterGuid == Prepared.ClusterGuid &&
        Component.OriginX == Resident.Origin.X &&
        Component.OriginY == Resident.Origin.Y &&
        Component.OriginZ == Resident.Origin.Z &&
        Component.InstanceCount == Resident.InstanceCount &&
        Component.PageCount == Resident.PageCount;
}

internal readonly struct VegetationCullingSelection
{
    internal VegetationCullingSelection(
        Guid clusterGuid,
        ulong generation,
        Guid speciesGuid,
        int lodLevel,
        int visiblePageCount,
        int visibleInstanceCount,
        int budgetInstanceCount,
        double distanceSquared,
        double screenSpaceError,
        bool accepted)
    {
        ClusterGuid = clusterGuid;
        Generation = generation;
        SpeciesGuid = speciesGuid;
        LodLevel = lodLevel;
        VisiblePageCount = visiblePageCount;
        VisibleInstanceCount = visibleInstanceCount;
        BudgetInstanceCount = budgetInstanceCount;
        DistanceSquared = distanceSquared;
        ScreenSpaceError = screenSpaceError;
        Accepted = accepted;
    }

    public Guid ClusterGuid { get; }
    public ulong Generation { get; }
    public Guid SpeciesGuid { get; }
    public int LodLevel { get; }
    public int VisiblePageCount { get; }
    public int VisibleInstanceCount { get; }
    public int BudgetInstanceCount { get; }
    public double DistanceSquared { get; }
    public double ScreenSpaceError { get; }
    public bool Accepted { get; }
}

internal readonly record struct VegetationCullingMetrics(
    int SourceClusterCount,
    int CandidateSpeciesCount,
    int VisibleSpeciesCount,
    int CulledClusterCount,
    int CulledPageCount,
    int CulledSpeciesCount,
    int DroppedSpeciesCount,
    int SelectedSpeciesCount,
    int SelectedInstanceCount,
    int SelectedBatchCount,
    int MaximumSelectedLod)
{
    public bool Overflowed => DroppedSpeciesCount > 0;
}

internal sealed class VegetationCullingPlanner
{
    private const double MinimumDistance = 1.0e-6;
    private const float MaximumOriginRelativeCoordinate = 3.4028234e38f;

    private VegetationCullingCandidate[] m_CandidateStorage = Array.Empty<VegetationCullingCandidate>();
    private VegetationCullingPriority[] m_PriorityStorage = Array.Empty<VegetationCullingPriority>();
    private VegetationCullingHistory[] m_HistoryStorage = Array.Empty<VegetationCullingHistory>();
    private VegetationCullingSelection[] m_Output = Array.Empty<VegetationCullingSelection>();
    private int[] m_NodeStack = Array.Empty<int>();
    private bool[] m_VisiblePages = Array.Empty<bool>();
    private int m_CandidateCount;
    private int m_HistoryCount;
    private int m_OutputCount;

    public VegetationCullingMetrics Metrics { get; private set; }

    public ReadOnlySpan<VegetationCullingSelection> Plan(
        ReadOnlySpan<VegetationClusterCullingInput> inputs,
        in VegetationCullingView view,
        in VegetationCullingSettings settings)
    {
        m_CandidateCount = 0;
        m_OutputCount = 0;
        if (!view.IsValid || !settings.IsValid)
        {
            Metrics = new VegetationCullingMetrics(
                inputs.Length,
                0,
                0,
                0,
                0,
                0,
                inputs.Length,
                0,
                0,
                0,
                0);
            return ReadOnlySpan<VegetationCullingSelection>.Empty;
        }

        int culledClusterCount = 0;
        int culledPageCount = 0;
        int culledSpeciesCount = 0;
        for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            ref readonly VegetationClusterCullingInput input = ref inputs[inputIndex];
            if (!input.IsValid ||
                (input.Component.Flags & VegetationClusterFlags.Visible) == 0)
            {
                culledClusterCount++;
                continue;
            }

            if (!TryMarkVisiblePages(
                    input,
                    view,
                    settings,
                    ref culledPageCount))
            {
                culledClusterCount++;
                continue;
            }

            ReadOnlySpan<CookedVegetationSpecies> species = input.Prepared.Species;
            ReadOnlySpan<VegetationResidentSpecies> residentSpecies = input.Resident.Species;
            for (int speciesIndex = 0; speciesIndex < residentSpecies.Length; speciesIndex++)
            {
                VegetationResidentSpecies resident = residentSpecies[speciesIndex];
                if (!TryFindCookedSpecies(species, resident, out CookedVegetationSpecies descriptor))
                {
                    culledSpeciesCount++;
                    continue;
                }

                int visiblePageCount = 0;
                int visibleInstanceCount = 0;
                double distanceSquared = double.PositiveInfinity;
                float maximumRadius = 0.0f;
                ReadOnlySpan<VegetationResidentPageData> pages = input.Resident.Pages;
                for (int pageIndex = 0; pageIndex < pages.Length; pageIndex++)
                {
                    if (!m_VisiblePages[pageIndex] ||
                        !PageContainsSpecies(pages[pageIndex], resident.Guid))
                    {
                        continue;
                    }

                    VegetationResidentPageData page = pages[pageIndex];
                    visiblePageCount++;
                    distanceSquared = Math.Min(
                        distanceSquared,
                        DistanceSquared(view.CameraWorldPosition, page.Bounds));
                    if (TryFindPageSpeciesRange(
                            page,
                            resident,
                            out CookedVegetationSpeciesInstanceRange range))
                    {
                        visibleInstanceCount = checked(visibleInstanceCount + range.InstanceCount);
                        maximumRadius = Math.Max(
                            maximumRadius,
                            range.MaximumConservativeRadius);
                        distanceSquared = Math.Min(
                            distanceSquared,
                            DistanceSquared(view.CameraWorldPosition, range.Bounds));
                        continue;
                    }

                    ReadOnlySpan<VegetationResidentInstance> instances = page.Instances;
                    for (int instanceIndex = 0; instanceIndex < instances.Length; instanceIndex++)
                    {
                        ref readonly VegetationResidentInstance instance = ref instances[instanceIndex];
                        if (instance.SpeciesGuid != resident.Guid)
                        {
                            continue;
                        }

                        visibleInstanceCount++;
                        maximumRadius = Math.Max(maximumRadius, instance.ConservativeRadius);
                    }
                }

                if (visibleInstanceCount == 0 || visiblePageCount == 0)
                {
                    culledSpeciesCount++;
                    continue;
                }

                if (TryFindSpeciesAcceleration(
                        input.Prepared.Acceleration,
                        resident,
                        out CookedVegetationSpeciesAcceleration acceleration))
                {
                    maximumRadius = Math.Max(
                        maximumRadius,
                        acceleration.MaximumConservativeRadius);
                }
                double speciesDistanceSquared = double.IsFinite(distanceSquared)
                    ? distanceSquared
                    : DistanceSquared(view.CameraWorldPosition, input.Resident.Bounds);
                double screenSpaceError = CalculateScreenSpaceError(
                    maximumRadius,
                    speciesDistanceSquared,
                    view);
                int desiredLod = SelectLod(
                    descriptor,
                    speciesDistanceSquared,
                    screenSpaceError,
                    view,
                    settings,
                    FindPreviousLod(
                        input.Resident.Guid,
                        input.Resident.Generation,
                        resident.Guid));
                EnsureCandidateCapacity(m_CandidateCount + 1);
                m_CandidateStorage[m_CandidateCount++] = new VegetationCullingCandidate(
                    input.Resident.Guid,
                    input.Resident.Generation,
                    resident.Guid,
                    desiredLod,
                    visiblePageCount,
                    visibleInstanceCount,
                    ScaleForBudget(visibleInstanceCount, settings.DensityMultiplier),
                    speciesDistanceSquared,
                    screenSpaceError,
                    IsDensityAccepted(
                        input.Resident.Guid,
                        resident.Guid,
                        settings.DensityMultiplier));
            }
        }

        if (m_CandidateCount == 0)
        {
            m_HistoryCount = 0;
            Metrics = new VegetationCullingMetrics(
                inputs.Length,
                0,
                0,
                culledClusterCount,
                culledPageCount,
                culledSpeciesCount,
                0,
                0,
                0,
                0,
                0);
            return ReadOnlySpan<VegetationCullingSelection>.Empty;
        }

        SortCandidates(m_CandidateStorage, m_CandidateCount);
        SaveHistory();
        int visibleSpeciesCount = m_CandidateCount;
        SelectBudgetedCandidates(settings, out int droppedSpeciesCount);
        BuildOutput();

        int selectedSpeciesCount = 0;
        int selectedInstanceCount = 0;
        int maximumSelectedLod = 0;
        for (int index = 0; index < m_OutputCount; index++)
        {
            ref readonly VegetationCullingSelection selection = ref m_Output[index];
            if (!selection.Accepted)
            {
                continue;
            }

            selectedSpeciesCount++;
            selectedInstanceCount = checked(
                selectedInstanceCount + selection.BudgetInstanceCount);
            maximumSelectedLod = Math.Max(maximumSelectedLod, selection.LodLevel);
        }

        Metrics = new VegetationCullingMetrics(
            inputs.Length,
            m_CandidateCount,
            visibleSpeciesCount,
            culledClusterCount,
            culledPageCount,
            culledSpeciesCount,
            droppedSpeciesCount,
            selectedSpeciesCount,
            selectedInstanceCount,
            selectedSpeciesCount,
            maximumSelectedLod);
        return new ReadOnlySpan<VegetationCullingSelection>(m_Output, 0, m_OutputCount);
    }

    internal void Reset()
    {
        m_CandidateCount = 0;
        m_HistoryCount = 0;
        m_OutputCount = 0;
        Metrics = default;
    }

    private bool TryMarkVisiblePages(
        in VegetationClusterCullingInput input,
        in VegetationCullingView view,
        in VegetationCullingSettings settings,
        ref int culledPageCount)
    {
        if (!TryGetRelativeBounds(input.Resident.Bounds, view.RenderOrigin, out _, out _) ||
            (settings.EnableFrustumCulling &&
             !IsVisible(input.Resident.Bounds, view)) ||
            (settings.EnableDistanceCulling &&
             IsBeyondDistance(
                 input.Resident.Bounds,
                 view.CameraWorldPosition,
                 settings.MaximumDistance)))
        {
            culledPageCount = checked(culledPageCount + input.Resident.Pages.Length);
            return false;
        }

        ReadOnlySpan<VegetationResidentPageData> pages = input.Resident.Pages;
        EnsurePageCapacity(pages.Length);
        Array.Clear(m_VisiblePages, 0, pages.Length);
        int visiblePageCount = 0;
        CookedVegetationClusterAcceleration? acceleration = input.Prepared.Acceleration;
        if (acceleration == null || acceleration.SpatialNodes.Count == 0)
        {
            for (int pageIndex = 0; pageIndex < pages.Length; pageIndex++)
            {
                if (IsPageVisible(pages[pageIndex], view, settings))
                {
                    m_VisiblePages[pageIndex] = true;
                    visiblePageCount++;
                }
                else
                {
                    culledPageCount++;
                }
            }

            return visiblePageCount > 0;
        }

        EnsureNodeCapacity(acceleration.SpatialNodes.Count);
        int stackCount = 1;
        m_NodeStack[0] = 0;
        while (stackCount > 0)
        {
            int nodeIndex = m_NodeStack[--stackCount];
            CookedVegetationSpatialNode node = acceleration.SpatialNodes[nodeIndex];
            if ((settings.EnableFrustumCulling && !IsVisible(node.Bounds, view)) ||
                (settings.EnableDistanceCulling &&
                 IsBeyondDistance(node.Bounds, view.CameraWorldPosition, settings.MaximumDistance)))
            {
                culledPageCount = checked(culledPageCount + node.PageCount);
                continue;
            }

            if (node.IsLeaf)
            {
                int pageIndex = node.FirstPageIndex;
                if (pageIndex >= 0 && pageIndex < pages.Length &&
                    IsPageVisible(pages[pageIndex], view, settings) &&
                    !m_VisiblePages[pageIndex])
                {
                    m_VisiblePages[pageIndex] = true;
                    visiblePageCount++;
                }
                continue;
            }

            if (node.RightChildIndex >= 0)
            {
                m_NodeStack[stackCount++] = node.RightChildIndex;
            }
            if (node.LeftChildIndex >= 0)
            {
                m_NodeStack[stackCount++] = node.LeftChildIndex;
            }
        }

        for (int pageIndex = 0; pageIndex < pages.Length; pageIndex++)
        {
            if (!m_VisiblePages[pageIndex])
            {
                culledPageCount++;
            }
        }

        return visiblePageCount > 0;
    }

    private static bool IsPageVisible(
        VegetationResidentPageData page,
        in VegetationCullingView view,
        in VegetationCullingSettings settings) =>
        TryGetRelativeBounds(page.Bounds, view.RenderOrigin, out _, out _) &&
        (!settings.EnableFrustumCulling || IsVisible(page.Bounds, view)) &&
        (!settings.EnableDistanceCulling ||
         !IsBeyondDistance(page.Bounds, view.CameraWorldPosition, settings.MaximumDistance));

    private static bool TryGetRelativeBounds(
        in WorldBounds bounds,
        in WorldPosition renderOrigin,
        out Vector3 relativeMin,
        out Vector3 relativeMax)
    {
        relativeMin = default;
        relativeMax = default;
        double minX = bounds.Min.X - renderOrigin.X;
        double minY = bounds.Min.Y - renderOrigin.Y;
        double minZ = bounds.Min.Z - renderOrigin.Z;
        double maxX = bounds.Max.X - renderOrigin.X;
        double maxY = bounds.Max.Y - renderOrigin.Y;
        double maxZ = bounds.Max.Z - renderOrigin.Z;
        if (!double.IsFinite(minX) || !double.IsFinite(minY) || !double.IsFinite(minZ) ||
            !double.IsFinite(maxX) || !double.IsFinite(maxY) || !double.IsFinite(maxZ) ||
            Math.Abs(minX) > MaximumOriginRelativeCoordinate ||
            Math.Abs(minY) > MaximumOriginRelativeCoordinate ||
            Math.Abs(minZ) > MaximumOriginRelativeCoordinate ||
            Math.Abs(maxX) > MaximumOriginRelativeCoordinate ||
            Math.Abs(maxY) > MaximumOriginRelativeCoordinate ||
            Math.Abs(maxZ) > MaximumOriginRelativeCoordinate)
        {
            return false;
        }

        relativeMin = new Vector3((float)minX, (float)minY, (float)minZ);
        relativeMax = new Vector3((float)maxX, (float)maxY, (float)maxZ);
        return true;
    }

    private static bool IsVisible(
        in WorldBounds bounds,
        in VegetationCullingView view)
    {
        if (!TryGetRelativeBounds(
                bounds,
                view.RenderOrigin,
                out Vector3 minimum,
                out Vector3 maximum))
        {
            return false;
        }

        Vector3 center = (minimum + maximum) * 0.5f;
        Vector3 extents = Vector3.Abs((maximum - minimum) * 0.5f);
        Matrix4x4 matrix = view.OriginRelativeViewProjection;
        return !OutsidePlane(
                   new Vector4(matrix.M14 + matrix.M11, matrix.M24 + matrix.M21,
                       matrix.M34 + matrix.M31, matrix.M44 + matrix.M41),
                   center,
                   extents) &&
               !OutsidePlane(
                   new Vector4(matrix.M14 - matrix.M11, matrix.M24 - matrix.M21,
                       matrix.M34 - matrix.M31, matrix.M44 - matrix.M41),
                   center,
                   extents) &&
               !OutsidePlane(
                   new Vector4(matrix.M14 + matrix.M12, matrix.M24 + matrix.M22,
                       matrix.M34 + matrix.M32, matrix.M44 + matrix.M42),
                   center,
                   extents) &&
               !OutsidePlane(
                   new Vector4(matrix.M14 - matrix.M12, matrix.M24 - matrix.M22,
                       matrix.M34 - matrix.M32, matrix.M44 - matrix.M42),
                   center,
                   extents) &&
               !OutsidePlane(
                   new Vector4(matrix.M13, matrix.M23, matrix.M33, matrix.M43),
                   center,
                   extents) &&
               !OutsidePlane(
                   new Vector4(matrix.M14 - matrix.M13, matrix.M24 - matrix.M23,
                       matrix.M34 - matrix.M33, matrix.M44 - matrix.M43),
                   center,
                   extents);
    }

    private static bool OutsidePlane(Vector4 plane, Vector3 center, Vector3 extents)
    {
        float distance =
            plane.X * center.X + plane.Y * center.Y + plane.Z * center.Z + plane.W;
        float radius =
            MathF.Abs(plane.X) * extents.X +
            MathF.Abs(plane.Y) * extents.Y +
            MathF.Abs(plane.Z) * extents.Z;
        return distance + radius < 0.0f;
    }

    private static bool IsBeyondDistance(
        in WorldBounds bounds,
        in WorldPosition camera,
        double maximumDistance)
    {
        double distanceSquared = DistanceSquared(camera, bounds);
        return distanceSquared > maximumDistance * maximumDistance;
    }

    private static double DistanceSquared(
        in WorldPosition camera,
        in WorldBounds bounds)
    {
        double x = AxisDistance(camera.X, bounds.Min.X, bounds.Max.X);
        double y = AxisDistance(camera.Y, bounds.Min.Y, bounds.Max.Y);
        double z = AxisDistance(camera.Z, bounds.Min.Z, bounds.Max.Z);
        return (x * x) + (y * y) + (z * z);
    }

    private static double AxisDistance(double value, double minimum, double maximum) =>
        value < minimum
            ? minimum - value
            : value > maximum
                ? value - maximum
                : 0.0;

    private static bool PageContainsSpecies(
        VegetationResidentPageData page,
        Guid speciesGuid)
    {
        ReadOnlySpan<VegetationResidentSpecies> species = page.Species;
        for (int index = 0; index < species.Length; index++)
        {
            if (species[index].Guid == speciesGuid)
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryFindPageSpeciesRange(
        VegetationResidentPageData page,
        in VegetationResidentSpecies species,
        out CookedVegetationSpeciesInstanceRange range)
    {
        CookedVegetationInstancePageAcceleration? acceleration = page.Acceleration;
        if (acceleration != null)
        {
            ReadOnlySpan<VegetationResidentSpecies> pageSpecies = page.Species;
            IReadOnlyList<CookedVegetationSpeciesInstanceRange> ranges =
                acceleration.SpeciesRanges;
            for (int index = 0; index < ranges.Count; index++)
            {
                CookedVegetationSpeciesInstanceRange candidate = ranges[index];
                int speciesIndex = checked((int)candidate.SpeciesIndex);
                if ((uint)speciesIndex < (uint)pageSpecies.Length &&
                    pageSpecies[speciesIndex].Guid == species.Guid &&
                    string.Equals(
                        pageSpecies[speciesIndex].PackageId,
                        species.PackageId,
                        StringComparison.Ordinal))
                {
                    range = candidate;
                    return true;
                }
            }
        }

        range = null!;
        return false;
    }

    private static bool TryFindCookedSpecies(
        ReadOnlySpan<CookedVegetationSpecies> species,
        in VegetationResidentSpecies resident,
        out CookedVegetationSpecies descriptor)
    {
        for (int index = 0; index < species.Length; index++)
        {
            CookedVegetationSpecies candidate = species[index];
            if (candidate.Guid == resident.Guid &&
                string.Equals(candidate.PackageId, resident.PackageId, StringComparison.Ordinal))
            {
                descriptor = candidate;
                return true;
            }
        }

        descriptor = null!;
        return false;
    }

    private static bool TryFindSpeciesAcceleration(
        CookedVegetationClusterAcceleration? acceleration,
        in VegetationResidentSpecies species,
        out CookedVegetationSpeciesAcceleration result)
    {
        if (acceleration != null)
        {
            for (int index = 0; index < acceleration.Species.Count; index++)
            {
                CookedVegetationSpeciesAcceleration candidate = acceleration.Species[index];
                if (candidate.Species.Guid == species.Guid &&
                    string.Equals(
                        candidate.Species.PackageId,
                        species.PackageId,
                        StringComparison.Ordinal))
                {
                    result = candidate;
                    return true;
                }
            }
        }

        result = null!;
        return false;
    }

    private static int SelectLod(
        CookedVegetationSpecies species,
        double distanceSquared,
        double screenSpaceError,
        in VegetationCullingView view,
        in VegetationCullingSettings settings,
        int previousLod)
    {
        int maximumLod = species.Lods.Count - 1;
        int desired = 0;
        double distance = Math.Sqrt(Math.Max(0.0, distanceSquared));
        while (desired < maximumLod)
        {
            CookedVegetationSpeciesLod lod = species.Lods[desired];
            bool beyondDistance = lod.MaximumDistance > 0.0f &&
                distance > lod.MaximumDistance;
            bool belowScreenError = lod.MaximumScreenError > 0.0f &&
                screenSpaceError < lod.MaximumScreenError;
            if (!beyondDistance && !belowScreenError)
            {
                break;
            }

            desired++;
        }

        if (previousLod < 0 || previousLod > maximumLod)
        {
            return desired;
        }

        int selected = previousLod;
        double hysteresis = settings.HysteresisFraction;
        if (desired > previousLod)
        {
            CookedVegetationSpeciesLod boundary = species.Lods[previousLod];
            double distanceBoundary = boundary.MaximumDistance;
            double errorBoundary = boundary.MaximumScreenError;
            bool stillWithinBand =
                (distanceBoundary > 0.0 &&
                 distance < distanceBoundary * (1.0 + hysteresis)) ||
                (errorBoundary > 0.0 &&
                 screenSpaceError > errorBoundary * (1.0 - hysteresis));
            if (!stillWithinBand)
            {
                selected = desired;
            }
        }
        else if (desired < previousLod)
        {
            CookedVegetationSpeciesLod boundary = species.Lods[desired];
            double distanceBoundary = boundary.MaximumDistance;
            double errorBoundary = boundary.MaximumScreenError;
            bool crossedFineBand =
                (distanceBoundary > 0.0 &&
                 distance < distanceBoundary * (1.0 - hysteresis)) ||
                (errorBoundary > 0.0 &&
                 screenSpaceError > errorBoundary * (1.0 + hysteresis));
            if (crossedFineBand)
            {
                selected = desired;
            }
        }

        return selected;
    }

    private static double CalculateScreenSpaceError(
        double radius,
        double distanceSquared,
        in VegetationCullingView view)
    {
        double projected = Math.Max(0.0, radius) * view.ProjectionScale;
        return view.Projection == VegetationCullingProjection.Perspective
            ? projected / Math.Max(MinimumDistance, Math.Sqrt(Math.Max(0.0, distanceSquared)))
            : projected;
    }

    private int FindPreviousLod(Guid clusterGuid, ulong generation, Guid speciesGuid)
    {
        int low = 0;
        int high = m_HistoryCount - 1;
        while (low <= high)
        {
            int middle = low + ((high - low) >> 1);
            ref readonly VegetationCullingHistory history = ref m_HistoryStorage[middle];
            int comparison = CompareIdentity(
                clusterGuid,
                generation,
                speciesGuid,
                history.ClusterGuid,
                history.Generation,
                history.SpeciesGuid);
            if (comparison == 0)
            {
                return history.LodLevel;
            }

            if (comparison < 0)
            {
                high = middle - 1;
            }
            else
            {
                low = middle + 1;
            }
        }

        return -1;
    }

    private void SaveHistory()
    {
        EnsureHistoryCapacity(m_CandidateCount);
        for (int index = 0; index < m_CandidateCount; index++)
        {
            VegetationCullingCandidate candidate = m_CandidateStorage[index];
            m_HistoryStorage[index] = new VegetationCullingHistory(
                candidate.ClusterGuid,
                candidate.Generation,
                candidate.SpeciesGuid,
                candidate.LodLevel);
        }

        m_HistoryCount = m_CandidateCount;
    }

    private void SelectBudgetedCandidates(
        in VegetationCullingSettings settings,
        out int droppedSpeciesCount)
    {
        for (int index = 0; index < m_CandidateCount; index++)
        {
            m_CandidateStorage[index].Accepted = false;
        }

        EnsurePriorityCapacity(m_CandidateCount);
        for (int index = 0; index < m_CandidateCount; index++)
        {
            VegetationCullingCandidate candidate = m_CandidateStorage[index];
            m_PriorityStorage[index] = new VegetationCullingPriority(
                index,
                candidate.DistanceSquared,
                candidate.ClusterGuid,
                candidate.SpeciesGuid);
        }

        SortPriorities(m_PriorityStorage, m_CandidateCount);
        int selectedInstances = 0;
        int selectedBatches = 0;
        droppedSpeciesCount = 0;
        for (int priorityIndex = 0; priorityIndex < m_CandidateCount; priorityIndex++)
        {
            int candidateIndex = m_PriorityStorage[priorityIndex].CandidateIndex;
            ref VegetationCullingCandidate candidate = ref m_CandidateStorage[candidateIndex];
            bool accepted = candidate.DensityAccepted &&
                selectedBatches < settings.MaximumBatchCount &&
                candidate.BudgetInstanceCount <=
                    settings.MaximumInstanceCount - selectedInstances;
            candidate.Accepted = accepted;
            if (!accepted)
            {
                droppedSpeciesCount++;
                continue;
            }

            selectedBatches++;
            selectedInstances = checked(selectedInstances + candidate.BudgetInstanceCount);
        }
    }

    private void BuildOutput()
    {
        EnsureOutputCapacity(m_CandidateCount);
        for (int index = 0; index < m_CandidateCount; index++)
        {
            ref readonly VegetationCullingCandidate candidate = ref m_CandidateStorage[index];
            m_Output[index] = new VegetationCullingSelection(
                candidate.ClusterGuid,
                candidate.Generation,
                candidate.SpeciesGuid,
                candidate.LodLevel,
                candidate.VisiblePageCount,
                candidate.VisibleInstanceCount,
                candidate.BudgetInstanceCount,
                candidate.DistanceSquared,
                candidate.ScreenSpaceError,
                candidate.Accepted);
        }

        m_OutputCount = m_CandidateCount;
    }

    private static bool IsDensityAccepted(Guid clusterGuid, Guid speciesGuid, double density)
    {
        if (density >= 1.0)
        {
            return true;
        }
        if (density <= 0.0)
        {
            return false;
        }

        Span<byte> bytes = stackalloc byte[32];
        clusterGuid.TryWriteBytes(bytes[..16], bigEndian: true, out _);
        speciesGuid.TryWriteBytes(bytes[16..], bigEndian: true, out _);
        ulong hash = 1469598103934665603UL;
        for (int index = 0; index < bytes.Length; index++)
        {
            hash ^= bytes[index];
            hash *= 1099511628211UL;
        }

        return (hash >> 11) * (1.0 / 9_007_199_254_740_992.0) < density;
    }

    private static int ScaleForBudget(int count, double density)
    {
        if (density >= 1.0)
        {
            return count;
        }

        return Math.Max(1, checked((int)Math.Ceiling(count * density)));
    }

    private void EnsureCandidateCapacity(int required)
    {
        if (required <= m_CandidateStorage.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_CandidateStorage.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_CandidateStorage, capacity);
    }

    private void EnsurePriorityCapacity(int required)
    {
        if (required <= m_PriorityStorage.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_PriorityStorage.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_PriorityStorage, capacity);
    }

    private void EnsureHistoryCapacity(int required)
    {
        if (required <= m_HistoryStorage.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_HistoryStorage.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_HistoryStorage, capacity);
    }

    private void EnsureOutputCapacity(int required)
    {
        if (required <= m_Output.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_Output.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_Output, capacity);
    }

    private void EnsureNodeCapacity(int required)
    {
        if (required <= m_NodeStack.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_NodeStack.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_NodeStack, capacity);
    }

    private void EnsurePageCapacity(int required)
    {
        if (required <= m_VisiblePages.Length)
        {
            return;
        }

        int capacity = Math.Max(4, m_VisiblePages.Length);
        while (capacity < required)
        {
            capacity = checked(capacity * 2);
        }
        Array.Resize(ref m_VisiblePages, capacity);
    }

    private static void SortCandidates(
        VegetationCullingCandidate[] candidates,
        int count)
    {
        for (int root = (count >> 1) - 1; root >= 0; root--)
        {
            SiftDownCandidates(candidates, root, count);
        }
        for (int end = count - 1; end > 0; end--)
        {
            (candidates[0], candidates[end]) = (candidates[end], candidates[0]);
            SiftDownCandidates(candidates, 0, end);
        }
    }

    private static void SiftDownCandidates(
        VegetationCullingCandidate[] candidates,
        int root,
        int count)
    {
        while (true)
        {
            int child = checked((root << 1) + 1);
            if (child >= count)
            {
                return;
            }

            int right = child + 1;
            if (right < count && CompareCandidates(candidates[child], candidates[right]) < 0)
            {
                child = right;
            }
            if (CompareCandidates(candidates[root], candidates[child]) >= 0)
            {
                return;
            }

            (candidates[root], candidates[child]) = (candidates[child], candidates[root]);
            root = child;
        }
    }

    private static int CompareCandidates(
        in VegetationCullingCandidate left,
        in VegetationCullingCandidate right) =>
        CompareIdentity(
            left.ClusterGuid,
            left.Generation,
            left.SpeciesGuid,
            right.ClusterGuid,
            right.Generation,
            right.SpeciesGuid);

    private static void SortPriorities(
        VegetationCullingPriority[] priorities,
        int count)
    {
        for (int root = (count >> 1) - 1; root >= 0; root--)
        {
            SiftDownPriorities(priorities, root, count);
        }
        for (int end = count - 1; end > 0; end--)
        {
            (priorities[0], priorities[end]) = (priorities[end], priorities[0]);
            SiftDownPriorities(priorities, 0, end);
        }
    }

    private static void SiftDownPriorities(
        VegetationCullingPriority[] priorities,
        int root,
        int count)
    {
        while (true)
        {
            int child = checked((root << 1) + 1);
            if (child >= count)
            {
                return;
            }

            int right = child + 1;
            if (right < count && ComparePriorities(priorities[child], priorities[right]) < 0)
            {
                child = right;
            }
            if (ComparePriorities(priorities[root], priorities[child]) >= 0)
            {
                return;
            }

            (priorities[root], priorities[child]) = (priorities[child], priorities[root]);
            root = child;
        }
    }

    private static int ComparePriorities(
        in VegetationCullingPriority left,
        in VegetationCullingPriority right)
    {
        int result = left.DistanceSquared.CompareTo(right.DistanceSquared);
        return result != 0
            ? result
            : CompareIdentity(
                left.ClusterGuid,
                0,
                left.SpeciesGuid,
                right.ClusterGuid,
                0,
                right.SpeciesGuid);
    }

    private static int CompareIdentity(
        Guid leftCluster,
        ulong leftGeneration,
        Guid leftSpecies,
        Guid rightCluster,
        ulong rightGeneration,
        Guid rightSpecies)
    {
        int result = leftCluster.CompareTo(rightCluster);
        if (result != 0)
        {
            return result;
        }

        result = leftGeneration.CompareTo(rightGeneration);
        return result != 0 ? result : leftSpecies.CompareTo(rightSpecies);
    }

    private struct VegetationCullingCandidate
    {
        public VegetationCullingCandidate(
            Guid clusterGuid,
            ulong generation,
            Guid speciesGuid,
            int lodLevel,
            int visiblePageCount,
            int visibleInstanceCount,
            int budgetInstanceCount,
            double distanceSquared,
            double screenSpaceError,
            bool densityAccepted)
        {
            ClusterGuid = clusterGuid;
            Generation = generation;
            SpeciesGuid = speciesGuid;
            LodLevel = lodLevel;
            VisiblePageCount = visiblePageCount;
            VisibleInstanceCount = visibleInstanceCount;
            BudgetInstanceCount = budgetInstanceCount;
            DistanceSquared = distanceSquared;
            ScreenSpaceError = screenSpaceError;
            DensityAccepted = densityAccepted;
            Accepted = false;
        }

        public Guid ClusterGuid;
        public ulong Generation;
        public Guid SpeciesGuid;
        public int LodLevel;
        public int VisiblePageCount;
        public int VisibleInstanceCount;
        public int BudgetInstanceCount;
        public double DistanceSquared;
        public double ScreenSpaceError;
        public bool DensityAccepted;
        public bool Accepted;
    }

    private readonly record struct VegetationCullingPriority(
        int CandidateIndex,
        double DistanceSquared,
        Guid ClusterGuid,
        Guid SpeciesGuid);

    private readonly record struct VegetationCullingHistory(
        Guid ClusterGuid,
        ulong Generation,
        Guid SpeciesGuid,
        int LodLevel);
}
