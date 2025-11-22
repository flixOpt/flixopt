"""
Test backward compatibility and deprecation warnings for InvestParameters.

This test verifies that:
1. Old parameter names (fix_effects, specific_effects, divest_effects, piecewise_effects) still work with warnings
2. New parameter names (effects_of_investment, effects_of_investment_per_size, effects_of_retirement, piecewise_effects_of_investment) work correctly
3. Both old and new approaches produce equivalent results
"""

import warnings

import pytest

from flixopt.interface import InvestParameters


class TestInvestParametersDeprecation:
    """Test suite for InvestParameters parameter deprecation."""

    def test_new_parameters_no_warnings(self):
        """Test that new parameter names don't trigger warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # Should not raise DeprecationWarning
            params = InvestParameters(
                fixed_size=100,
                effects_of_investment={'cost': 25000},
                effects_of_investment_per_size={'cost': 1200},
                effects_of_retirement={'cost': 5000},
            )
            assert params.effects_of_investment == {'cost': 25000}
            assert params.effects_of_investment_per_size == {'cost': 1200}
            assert params.effects_of_retirement == {'cost': 5000}

    def test_old_fix_effects_deprecation_warning(self):
        """Test that fix_effects triggers deprecation warning."""
        with pytest.warns(DeprecationWarning, match='fix_effects.*deprecated.*effects_of_investment'):
            params = InvestParameters(fix_effects={'cost': 25000})
            # Verify backward compatibility
            assert params.effects_of_investment == {'cost': 25000}

        # Accessing the property also triggers warning
        with pytest.warns(DeprecationWarning, match='fix_effects.*deprecated.*effects_of_investment'):
            assert params.fix_effects == {'cost': 25000}

    def test_old_specific_effects_deprecation_warning(self):
        """Test that specific_effects triggers deprecation warning."""
        with pytest.warns(DeprecationWarning, match='specific_effects.*deprecated.*effects_of_investment_per_size'):
            params = InvestParameters(specific_effects={'cost': 1200})
            # Verify backward compatibility
            assert params.effects_of_investment_per_size == {'cost': 1200}

        # Accessing the property also triggers warning
        with pytest.warns(DeprecationWarning, match='specific_effects.*deprecated.*effects_of_investment_per_size'):
            assert params.specific_effects == {'cost': 1200}

    def test_old_divest_effects_deprecation_warning(self):
        """Test that divest_effects triggers deprecation warning."""
        with pytest.warns(DeprecationWarning, match='divest_effects.*deprecated.*effects_of_retirement'):
            params = InvestParameters(divest_effects={'cost': 5000})
            # Verify backward compatibility
            assert params.effects_of_retirement == {'cost': 5000}

        # Accessing the property also triggers warning
        with pytest.warns(DeprecationWarning, match='divest_effects.*deprecated.*effects_of_retirement'):
            assert params.divest_effects == {'cost': 5000}

    def test_old_piecewise_effects_deprecation_warning(self):
        """Test that piecewise_effects triggers deprecation warning."""
        from flixopt.interface import Piece, Piecewise, PiecewiseEffects

        test_piecewise = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 100)]),
            piecewise_shares={'cost': Piecewise([Piece(800, 600)])},
        )
        with pytest.warns(DeprecationWarning, match='piecewise_effects.*deprecated.*piecewise_effects_of_investment'):
            params = InvestParameters(piecewise_effects=test_piecewise)
            # Verify backward compatibility
            assert params.piecewise_effects_of_investment is test_piecewise

        # Accessing the property also triggers warning
        with pytest.warns(DeprecationWarning, match='piecewise_effects.*deprecated.*piecewise_effects_of_investment'):
            assert params.piecewise_effects is test_piecewise

    def test_all_old_parameters_together(self):
        """Test all old parameters work together with warnings."""
        from flixopt.interface import Piece, Piecewise, PiecewiseEffects

        test_piecewise = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 100)]),
            piecewise_shares={'cost': Piecewise([Piece(800, 600)])},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', DeprecationWarning)
            params = InvestParameters(
                fixed_size=100,
                fix_effects={'cost': 25000},
                specific_effects={'cost': 1200},
                divest_effects={'cost': 5000},
                piecewise_effects=test_piecewise,
            )

            # Should trigger 4 deprecation warnings (from kwargs)
            assert len([warning for warning in w if issubclass(warning.category, DeprecationWarning)]) == 4

            # Verify all mappings work (accessing new properties - no warnings)
            assert params.effects_of_investment == {'cost': 25000}
            assert params.effects_of_investment_per_size == {'cost': 1200}
            assert params.effects_of_retirement == {'cost': 5000}
            assert params.piecewise_effects_of_investment is test_piecewise

        # Verify old attributes still work (accessing deprecated properties - triggers warnings)
        with pytest.warns(DeprecationWarning):
            assert params.fix_effects == {'cost': 25000}
        with pytest.warns(DeprecationWarning):
            assert params.specific_effects == {'cost': 1200}
        with pytest.warns(DeprecationWarning):
            assert params.divest_effects == {'cost': 5000}
        with pytest.warns(DeprecationWarning):
            assert params.piecewise_effects is test_piecewise

    def test_both_old_and_new_raises_error(self):
        """Test that specifying both old and new parameter names raises ValueError."""
        # fix_effects + effects_of_investment
        with pytest.raises(
            ValueError, match='Either fix_effects or effects_of_investment can be specified, but not both'
        ):
            InvestParameters(
                fix_effects={'cost': 10000},
                effects_of_investment={'cost': 25000},
            )

        # specific_effects + effects_of_investment_per_size
        with pytest.raises(
            ValueError,
            match='Either specific_effects or effects_of_investment_per_size can be specified, but not both',
        ):
            InvestParameters(
                specific_effects={'cost': 1200},
                effects_of_investment_per_size={'cost': 1500},
            )

        # divest_effects + effects_of_retirement
        with pytest.raises(
            ValueError, match='Either divest_effects or effects_of_retirement can be specified, but not both'
        ):
            InvestParameters(
                divest_effects={'cost': 5000},
                effects_of_retirement={'cost': 6000},
            )

        # piecewise_effects + piecewise_effects_of_investment
        from flixopt.interface import Piece, Piecewise, PiecewiseEffects

        test_piecewise1 = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 100)]),
            piecewise_shares={'cost': Piecewise([Piece(800, 600)])},
        )
        test_piecewise2 = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 200)]),
            piecewise_shares={'cost': Piecewise([Piece(900, 700)])},
        )
        with pytest.raises(
            ValueError,
            match='Either piecewise_effects or piecewise_effects_of_investment can be specified, but not both',
        ):
            InvestParameters(
                piecewise_effects=test_piecewise1,
                piecewise_effects_of_investment=test_piecewise2,
            )

    def test_piecewise_effects_of_investment_new_parameter(self):
        """Test that piecewise_effects_of_investment works correctly."""
        from flixopt.interface import Piece, Piecewise, PiecewiseEffects

        test_piecewise = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 100)]),
            piecewise_shares={'cost': Piecewise([Piece(800, 600)])},
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # Should not raise DeprecationWarning when using new parameter
            params = InvestParameters(piecewise_effects_of_investment=test_piecewise)
            assert params.piecewise_effects_of_investment is test_piecewise

        # Accessing deprecated property triggers warning
        with pytest.warns(DeprecationWarning):
            assert params.piecewise_effects is test_piecewise

    def test_backward_compatibility_with_features(self):
        """Test that old attribute names remain accessible for features.py compatibility."""
        from flixopt.interface import Piece, Piecewise, PiecewiseEffects

        test_piecewise = PiecewiseEffects(
            piecewise_origin=Piecewise([Piece(0, 100)]),
            piecewise_shares={'cost': Piecewise([Piece(800, 600)])},
        )

        params = InvestParameters(
            effects_of_investment={'cost': 25000},
            effects_of_investment_per_size={'cost': 1200},
            effects_of_retirement={'cost': 5000},
            piecewise_effects_of_investment=test_piecewise,
        )

        # Old properties should still be accessible (for features.py) but with warnings
        with pytest.warns(DeprecationWarning):
            assert params.fix_effects == {'cost': 25000}
        with pytest.warns(DeprecationWarning):
            assert params.specific_effects == {'cost': 1200}
        with pytest.warns(DeprecationWarning):
            assert params.divest_effects == {'cost': 5000}
        with pytest.warns(DeprecationWarning):
            assert params.piecewise_effects is test_piecewise

        # Properties should return the same objects as the new attributes
        with pytest.warns(DeprecationWarning):
            assert params.fix_effects is params.effects_of_investment
        with pytest.warns(DeprecationWarning):
            assert params.specific_effects is params.effects_of_investment_per_size
        with pytest.warns(DeprecationWarning):
            assert params.divest_effects is params.effects_of_retirement
        with pytest.warns(DeprecationWarning):
            assert params.piecewise_effects is params.piecewise_effects_of_investment

    def test_empty_parameters(self):
        """Test that empty/None parameters work correctly."""
        params = InvestParameters()

        assert params.effects_of_investment == {}
        assert params.effects_of_investment_per_size == {}
        assert params.effects_of_retirement == {}
        assert params.piecewise_effects_of_investment is None

        # Old properties should also be empty (but with warnings)
        with pytest.warns(DeprecationWarning):
            assert params.fix_effects == {}
        with pytest.warns(DeprecationWarning):
            assert params.specific_effects == {}
        with pytest.warns(DeprecationWarning):
            assert params.divest_effects == {}
        with pytest.warns(DeprecationWarning):
            assert params.piecewise_effects is None

    def test_mixed_old_and_new_parameters(self):
        """Test mixing old and new parameter names (not recommended but should work)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', DeprecationWarning)
            params = InvestParameters(
                effects_of_investment={'cost': 25000},  # New
                specific_effects={'cost': 1200},  # Old
                effects_of_retirement={'cost': 5000},  # New
            )

            # Should only warn about the old parameter
            assert len([warning for warning in w if issubclass(warning.category, DeprecationWarning)]) == 1

            # All should work correctly
            assert params.effects_of_investment == {'cost': 25000}
            assert params.effects_of_investment_per_size == {'cost': 1200}
            assert params.effects_of_retirement == {'cost': 5000}

    def test_unexpected_keyword_arguments(self):
        """Test that unexpected keyword arguments raise TypeError."""
        # Single unexpected argument
        with pytest.raises(
            TypeError, match="InvestParameters.__init__\\(\\) got unexpected keyword argument\\(s\\): 'invalid_param'"
        ):
            InvestParameters(invalid_param='value')

        # Multiple unexpected arguments
        with pytest.raises(
            TypeError,
            match="InvestParameters.__init__\\(\\) got unexpected keyword argument\\(s\\): 'param1', 'param2'",
        ):
            InvestParameters(param1='value1', param2='value2')

        # Mix of valid and invalid arguments
        with pytest.raises(
            TypeError, match="InvestParameters.__init__\\(\\) got unexpected keyword argument\\(s\\): 'typo'"
        ):
            InvestParameters(effects_of_investment={'cost': 100}, typo='value')

    def test_optional_parameter_deprecation(self):
        """Test that optional parameter triggers deprecation warning and maps to mandatory."""
        # Test optional=True (should map to mandatory=False)
        with pytest.warns(DeprecationWarning, match='optional.*deprecated.*mandatory'):
            params = InvestParameters(optional=True)
            assert params.mandatory is False

        # Test optional=False (should map to mandatory=True)
        with pytest.warns(DeprecationWarning, match='optional.*deprecated.*mandatory'):
            params = InvestParameters(optional=False)
            assert params.mandatory is True

    def test_mandatory_parameter_no_warning(self):
        """Test that mandatory parameter doesn't trigger warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # Test mandatory=True
            params = InvestParameters(mandatory=True)
            assert params.mandatory is True

            # Test mandatory=False (explicit)
            params = InvestParameters(mandatory=False)
            assert params.mandatory is False

    def test_mandatory_default_value(self):
        """Test that default value of mandatory is False when neither optional nor mandatory is specified."""
        params = InvestParameters()
        assert params.mandatory is False

    def test_both_optional_and_mandatory_no_error(self):
        """Test that specifying both optional and mandatory doesn't raise error.

        Note: Conflict checking is disabled for mandatory/optional because mandatory has
        a non-None default value (False), making it impossible to distinguish between
        an explicit mandatory=False and the default value. The deprecated optional
        parameter will take precedence when both are specified.
        """
        # When both are specified, optional takes precedence (with deprecation warning)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            params = InvestParameters(optional=True, mandatory=False)
            # optional=True should result in mandatory=False
            assert params.mandatory is False

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            params = InvestParameters(optional=False, mandatory=True)
            # optional=False should result in mandatory=True (optional takes precedence)
            assert params.mandatory is True

    def test_optional_property_deprecation(self):
        """Test that accessing optional property triggers deprecation warning."""
        params = InvestParameters(mandatory=True)

        # Reading the property triggers warning
        with pytest.warns(DeprecationWarning, match="Property 'optional' is deprecated"):
            assert params.optional is False

        # Setting the property triggers warning
        with pytest.warns(DeprecationWarning, match="Property 'optional' is deprecated"):
            params.optional = True
        assert params.mandatory is False
