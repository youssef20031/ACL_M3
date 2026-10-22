import { Check, X } from 'lucide-react';
import type { ExplainFixture } from '../../services/api';
import { SignedValue } from './Waterfall';

/*
 * The features that both SHAP and LIME rank among their 10 biggest for one fixture. Their sizes
 * aren't comparable (SHAP values add up to the prediction, LIME weights are a local linear
 * model's), so the column to read is whether they push the same way.
 */

interface ShapVsLimeProps {
  fixture: ExplainFixture;
  top?: number;
}

export function ShapVsLime({ fixture, top = 10 }: ShapVsLimeProps) {
  const shapTop = [...fixture.contributions].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, top);
  const limeTop = [...fixture.lime.weights].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, top);
  const limeByFeature = new Map<string, (typeof limeTop)[number]>();
  for (const w of limeTop) if (!limeByFeature.has(w.feature)) limeByFeature.set(w.feature, w);
  const rows = shapTop.flatMap((c) => {
    const w = limeByFeature.get(c.feature);
    return w ? [{ feature: c.feature, label: c.label, shap: c.shap, weight: w.weight }] : [];
  });

  if (!rows.length) {
    return <p className="text-data text-fg-subtle">SHAP and LIME share none of their top {top} features for this fixture.</p>;
  }
  const agree = rows.filter((r) => Math.sign(r.shap) === Math.sign(r.weight)).length;

  return (
    <div className="min-w-0">
      <p className="mb-2 text-data text-fg-muted">
        {rows.length} {rows.length === 1 ? 'feature is' : 'features are'} in both top {top}s; they push the same way for {agree} of {rows.length}.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-data">
          <caption className="sr-only">Features in both SHAP’s and LIME’s top {top}, with each method’s value in points</caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Feature</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">SHAP</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">LIME</th>
              <th scope="col" className="py-1.5 text-right font-semibold">
                <span className="hidden sm:inline">Same direction</span>
                <span className="sm:hidden">Agree</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const same = Math.sign(r.shap) === Math.sign(r.weight);
              return (
                <tr key={r.feature} className="border-t border-line">
                  <th scope="row" className="max-w-[10rem] truncate py-1.5 pr-3 font-normal text-fg sm:max-w-none" title={r.label}>
                    {r.label}
                  </th>
                  <td className="py-1.5 pr-3 text-right tabular-nums">
                    <SignedValue value={r.shap} />
                  </td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">
                    <SignedValue value={r.weight} />
                  </td>
                  <td className="py-1.5 text-right">
                    {same ? (
                      <Check className="ml-auto h-4 w-4 text-accent" aria-hidden="true" />
                    ) : (
                      <X className="ml-auto h-4 w-4 text-warning" aria-hidden="true" />
                    )}
                    <span className="sr-only">{same ? 'Yes' : 'No'}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
